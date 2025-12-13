import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import Draw # For ReactionToImage
import io
import base64
from operator import itemgetter
from itertools import combinations
import mols2grid
import seaborn as sns
import matplotlib.pyplot as plt
import subprocess
import sys

# --- Install necessary packages ---
# This block ensures all required libraries are installed when the script runs.
# 'rdkit-pypi' is used for pip installation of RDKit.
def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        st.success(f"Successfully installed {package}")
    except Exception as e:
        st.error(f"Failed to install {package}: {e}. Please install manually if the app fails to load.")

install_package("streamlit")
install_package("pandas")
install_package("rdkit-pypi") 
install_package("mols2grid")
install_package("seaborn")
install_package("matplotlib")


# --- Custom functions from previous notebook steps ---

# From useful_rdkit_utils logic
def get_largest_fragment(mol):
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True)
    if not frags:
        return None
    num_atoms = [x.GetNumHeavyAtoms() for x in frags]
    idx = num_atoms.index(max(num_atoms))
    return frags[idx]

# Helper for sorting fragments and removing map numbers
def remove_map_nums(mol):
    if mol is None: return
    for atm in mol.GetAtoms():
        atm.SetAtomMapNum(0)

def sort_fragments(mol):
    if mol is None: return []
    frag_list = list(Chem.GetMolFrags(mol, asMols=True))
    [remove_map_nums(x) for x in frag_list]
    frag_num_atoms_list = [(x.GetNumAtoms(), x) for x in frag_list]
    frag_num_atoms_list.sort(key=itemgetter(0), reverse=True)
    return [x[1] for x in frag_num_atoms_list]

# From scaffold_finder.py
def GetRingSystems(mol, includeSpiro=True):
    ri = mol.GetRingInfo()
    systems = []
    for ring in ri.AtomRings():
        ringAts = set(ring)
        sys = set(ringAts)
        for otherRing in ri.AtomRings():
            if ringAts!=set(otherRing) and len(ringAts.intersection(otherRing))>0:
                sys = sys.union(otherRing)
        systems.append(tuple(sorted(list(sys))))
    systems.sort(key=lambda x:len(x),reverse=True)
    res = []
    seen = [False]*len(systems)
    for i,sysI in enumerate(systems):
        if seen[i]:
            continue
        res.append(sysI)
        for j in range(i+1,len(systems)):
            if not seen[j] and len(set(sysI).intersection(set(systems[j])))>0:
                seen[j]=True
    return res

def FragmentMol(mol,maxCuts=3,minMolSize=1,linker_smarts="[D3-D4;!R]"):
    # remove stereochemistry for fragmentation
    clean_mol = Chem.Mol(mol)
    Chem.RemoveStereochemistry(clean_mol)
    clean_mol = Chem.AddHs(clean_mol)
    
    # if maxCuts is 0, just return the original molecule
    if maxCuts == 0:
        return [(0,clean_mol)]

    res = []
    # find all possible linker cut sites
    patt = Chem.MolFromSmarts(linker_smarts)
    matches = clean_mol.GetSubstructMatches(patt)
    cut_bonds = set()
    for m in matches:
        for atm_idx in m:
            atm = clean_mol.GetAtomWithIdx(atm_idx)
            for bond in atm.GetBonds():
                if not bond.IsInRing(): # exclude cutting bonds in a ring
                    cut_bonds.add(tuple(sorted((bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()))))

    if not cut_bonds:
        return [(0,clean_mol)]

    # try all combinations of cuts up to maxCuts
    for num_cuts in range(1,maxCuts+1):
        if num_cuts > len(cut_bonds):
            continue
        for combo_bonds in combinations(cut_bonds,num_cuts):
            bonds_to_cut = []
            for b_atm1, b_atm2 in combo_bonds:
                bond = clean_mol.GetBondBetweenAtoms(b_atm1, b_atm2)
                if bond:
                    bonds_to_cut.append(bond.GetIdx())
            
            if bonds_to_cut:
                tmp = Chem.FragmentOnBonds(clean_mol,bonds_to_cut,addDummies=True)
                res.append((num_cuts,tmp))
    return res

# --- Image generation helper functions ---
def rxn_to_base64_image(rxn):
    if rxn is None: return ""
    try:
        # Use RDKit's Draw.ReactionToImage which returns a PIL Image
        img = Draw.ReactionToImage(rxn)
        img_buffer = io.BytesIO()
        img.save(img_buffer, format="PNG")
        im_text64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
        img_str = f"<img src='data:image/png;base64,{im_text64}' style='width: 300px; height: 150px;'/>"
        return img_str
    except Exception as e:
        return f"Error rendering reaction: {e}"

def stripplot_base64_image(dist):
    sns.set(rc={'figure.figsize': (3, 1)})
    sns.set_style('whitegrid')
    fig, ax = plt.subplots()
    sns.stripplot(x=dist, ax=ax)
    ax.axvline(0,ls="--",c="red")
    ax.set_xlim(-5,5)
    ax.set_yticks([]) 
    ax.set_xlabel("")
    s = io.BytesIO()
    plt.savefig(s, format='png', bbox_inches="tight")
    plt.close(fig)
    s = base64.b64encode(s.getvalue()).decode("utf-8").replace("\n", "")
    return '<img align="left" src="data:image/png;base64,%s">' % s

def find_examples(delta_df, query_idx):
    example_list = []
    if query_idx is None:
        return pd.DataFrame(columns=["SMILES","Name","pIC50"])
    
    if 'idx' not in delta_df.columns:
        st.error("Error: 'idx' column not found in delta_df. Cannot find examples.")
        return pd.DataFrame(columns=["SMILES","Name","pIC50"])

    if not delta_df.empty and query_idx in delta_df['idx'].unique():
        for idx,row in delta_df.query("idx == @query_idx").sort_values("Delta",ascending=False).iterrows():
            smi_1, name_1, pIC50_1 = row.SMILES_1, row.Name_1, row.pIC50_1
            smi_2, name_2, pIC50_2 = row.SMILES_2, row.Name_2, row.pIC50_2
            tmp_list = [(smi_1, name_1, pIC50_1),(smi_2, name_2, pIC50_2)]
            tmp_list.sort(key=itemgetter(0)) 
            example_list.append(tmp_list[0])
            example_list.append(tmp_list[1])
    return pd.DataFrame(example_list,columns=["SMILES","Name","pIC50"])

# --- Streamlit App ---
def main():
    st.set_page_config(layout="wide")
    st.title("Matched Molecular Pairs (MMPs) Analysis")

    st.markdown("""
        This application performs Matched Molecular Pairs (MMPs) analysis on a dataset of chemical compounds.
        It identifies pairs of molecules that differ by a small chemical transformation and analyzes the
        impact of these transformations on a property (e.g., pIC50).
    """)

    # 1. Load Data
    st.header("1. Load Data")
    uploaded_file = st.file_uploader("Upload your CSV file (e.g., MMP_9_MMP.csv)", type="csv")
    
    df = None
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("Data loaded successfully!")
        st.dataframe(df.head())
    else:
        st.info("Please upload a CSV file to begin.")
        # Attempt to load from default path if in Colab context (for testing/convenience)
        if 'google.colab' in sys.modules:
            default_path = "/content/drive/MyDrive/MMP_9_MMP.csv"
            try:
                df = pd.read_csv(default_path)
                st.info(f"Using default file from Google Drive: `{default_path}`")
                st.dataframe(df.head())
            except FileNotFoundError:
                st.warning(f"Default file `{default_path}` not found in your Google Drive. Please upload manually or ensure it's in `/content/drive/MyDrive/`.")
            except Exception as e:
                st.error(f"Error loading default file from Drive: {e}")
                
    if df is None or df.empty:
        st.stop() # Stop execution if no data is loaded or data is empty

    if not all(col in df.columns for col in ['SMILES', 'Name', 'pIC50']):
        st.error("The CSV file must contain 'SMILES', 'Name', and 'pIC50' columns.")
        st.stop()

    # 2. Preprocessing
    st.header("2. Preprocessing Molecules")
    
    # Create RDKit Mol objects
    st.write("Creating RDKit molecule objects...")
    mols = []
    mol_from_smiles_bar = st.progress(0, text="Converting SMILES to Mol objects...")
    for i, smiles in enumerate(df.SMILES):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                mols.append(mol)
            else:
                mols.append(None)
        except:
            mols.append(None)
        mol_from_smiles_bar.progress((i + 1) / len(df))
    df['mol'] = mols
    mol_from_smiles_bar.empty()

    # Filter out rows where mol creation failed
    df_valid_mols = df.dropna(subset=['mol']).copy()
    if df_valid_mols.empty:
        st.error("No valid molecules could be parsed from SMILES. Please check your data.")
        st.stop()
    
    # Get largest fragment
    st.write("Getting largest fragments...")
    largest_frags = []
    largest_frag_bar = st.progress(0, text="Extracting largest fragments...")
    for i, mol in enumerate(df_valid_mols['mol']):
        if mol:
            largest_frags.append(get_largest_fragment(mol))
        else:
            largest_frags.append(None)
        largest_frag_bar.progress((i + 1) / len(df_valid_mols))
    df_valid_mols.loc[:, 'mol'] = largest_frags # Use .loc for setting values
    largest_frag_bar.empty()
    st.success("Molecules preprocessed successfully.")

    # 3. Decompose Molecules to Scaffolds and Sidechains
    st.header("3. Decompose Molecules")
    
    row_list = []
    decompose_progress = st.progress(0, text="Decomposing molecules...")
    for i, row_data in enumerate(df_valid_mols[['SMILES', 'Name', 'pIC50', 'mol']].itertuples(index=False)):
        smiles, name, pIC50, mol = row_data
        decompose_progress.progress((i + 1) / len(df_valid_mols), text=f"Decomposing molecule {i+1}/{len(df_valid_mols)}")
        if mol is None: continue 
        
        frag_list = FragmentMol(mol, maxCuts=1) # maxCuts=1 as per notebook
        for _, frag_mol in frag_list:
            if frag_mol is None: continue 
            pair_list = sort_fragments(frag_mol)
            
            core_smiles = Chem.MolToSmiles(pair_list[0]) if len(pair_list) > 0 and pair_list[0] else None
            r_group_smiles = Chem.MolToSmiles(pair_list[1]) if len(pair_list) > 1 and pair_list[1] else None

            if core_smiles: 
                row_list.append([smiles, core_smiles, r_group_smiles, name, pIC50])
                
    decompose_progress.empty()
    row_df = pd.DataFrame(row_list, columns=["SMILES", "Core", "R_group", "Name", "pIC50"])
    st.subheader("Decomposed Molecules (first 10 rows)")
    st.dataframe(row_df.head(10))

    # 4. Collect Pairs With the Same Scaffold
    st.header("4. Collect Matched Molecular Pairs")

    delta_list = []
    collect_pairs_progress = st.progress(0, text="Collecting pairs...")
    
    grouped_cores = list(row_df.groupby("Core"))
    for i, (k, v) in enumerate(grouped_cores):
        collect_pairs_progress.progress((i + 1) / len(grouped_cores), text=f"Processing core {i+1}/{len(grouped_cores)}")
        if len(v) >= 2: 
            for a, b in combinations(range(0, len(v)), 2):
                reagent_a = v.iloc[a]
                reagent_b = v.iloc[b]
                if str(reagent_a.SMILES) == str(reagent_b.SMILES):
                    continue
                reagent_a, reagent_b = sorted([reagent_a, reagent_b], key=lambda x: str(x.SMILES))

                # Handle cases where R_group might be None (e.g., if only a core was found)
                if reagent_a.R_group is None or reagent_b.R_group is None:
                    continue
                
                delta = reagent_b.pIC50 - reagent_a.pIC50
                transform_smarts = f"{str(reagent_a.R_group).replace('*','*-')}>>{str(reagent_b.R_group).replace('*','*-')}"
                
                delta_list.append(list(reagent_a.values) + list(reagent_b.values) + [transform_smarts, delta])
    
    collect_pairs_progress.empty()
    cols = ["SMILES_1","Core_1","R_group_1","Name_1","pIC50_1",
            "SMILES_2","Core_2","R_group_2","Name_2","pIC50_2", 
            "Transform","Delta"]
    delta_df = pd.DataFrame(delta_list, columns=cols)
    st.subheader("Delta DataFrame (last 10 rows)")
    st.dataframe(delta_df.tail(10))

    # 5. Collect Frequently Occurring Pairs
    st.header("5. Summarize Transformations")
    min_transform_occurrence = st.slider("Minimum transform occurrence:", 1, 20, 5)

    mmp_list = []
    for k, v in delta_df.groupby("Transform"):
        if len(v) >= min_transform_occurrence:
            mmp_list.append([k, len(v), list(v.Delta.values)]) 
    
    mmp_df = pd.DataFrame(mmp_list, columns=["Transform", "Count", "Deltas"])
    
    if mmp_df.empty:
        st.warning("No frequent transformations found with the current minimum occurrence. Try lowering the value.")
        st.stop()

    mmp_df['idx'] = range(0, len(mmp_df))
    mmp_df['mean_delta'] = [x.mean() for x in mmp_df.Deltas]
    
    # Calculate rxn_mol and image representations
    st.info("Generating reaction images and distribution plots. This may take a moment...")
    rxn_mol_bar = st.progress(0, text="Generating reaction diagrams...")
    mmp_df['rxn_mol'] = None 
    for i, row in mmp_df.iterrows():
        try:
            mmp_df.loc[i, 'rxn_mol'] = AllChem.ReactionFromSmarts(row.Transform.replace('*-','*'), useSmiles=True)
        except Exception: # Catch specific RDKit errors if possible
            mmp_df.loc[i, 'rxn_mol'] = None 
        rxn_mol_bar.progress((i + 1) / len(mmp_df))
    rxn_mol_bar.empty()

    # Filter out rows where rxn_mol could not be created before applying image functions
    mmp_df = mmp_df.dropna(subset=['rxn_mol']).copy()
    if mmp_df.empty:
        st.warning("No valid transformations could be converted to RDKit reaction objects. Cannot display summary.")
        st.stop()

    mmp_df['MMP Transform'] = mmp_df.rxn_mol.apply(rxn_to_base64_image)
    mmp_df['Delta Distribution'] = mmp_df.Deltas.apply(stripplot_base64_image)

    st.subheader("Frequent Transformations Summary")
    
    sort_by_column = st.selectbox(
        "Sort transformations by:",
        options=["mean_delta", "Count"],
        index=0
    )
    ascending_order = st.checkbox("Sort ascending", value=False if sort_by_column == "mean_delta" else True)
    
    mmp_df_display = mmp_df.sort_values(sort_by_column, ascending=ascending_order).copy() # Use copy() to avoid SettingWithCopyWarning
    
    cols_to_show = ['MMP Transform','Count',"mean_delta","Delta Distribution"]
    # Adjust column names for the HTML table
    html_df = mmp_df_display[cols_to_show].round(2)
    html_df.columns = ['MMP Transform', 'Count', 'Mean ΔpIC50', 'Delta Distribution']

    st.markdown(
        html_df.to_html(escape=False),
        unsafe_allow_html=True
    )
    
    # 6. Explore Specific Transformation Examples
    st.header("6. Explore Specific Transformation Examples")

    if not mmp_df_display.empty:
        selected_transform_idx = st.selectbox(
            "Select a transformation to view examples:",
            options=mmp_df_display['idx'].tolist(),
            format_func=lambda x: f"{mmp_df_display.loc[mmp_df_display['idx'] == x, 'Transform'].iloc[0]} (Mean Δ: {mmp_df_display.loc[mmp_df_display['idx'] == x, 'mean_delta'].iloc[0]:.2f})"
        )

        if selected_transform_idx is not None:
            selected_transform_row = mmp_df.loc[mmp_df['idx'] == selected_transform_idx].iloc[0]
            st.subheader(f"Details for Transformation: {selected_transform_row['Transform']}")
            st.markdown(f"**Mean ΔpIC50:** {selected_transform_row['mean_delta']:.2f}")
            st.markdown(f"**Occurrences:** {selected_transform_row['Count']}")
            
            st.markdown("---")
            st.write("Reaction Diagram:")
            st.markdown(selected_transform_row['MMP Transform'], unsafe_allow_html=True)
            
            st.write("ΔpIC50 Distribution:")
            st.markdown(selected_transform_row['Delta Distribution'], unsafe_allow_html=True)

            st.subheader("Example Compound Pairs:")
            
            example_df = find_examples(delta_df, selected_transform_idx)
            example_df_fixed = example_df.rename(columns={"SMILES": "ChEMBL_ID", "Name": "SMILES"})

            if not example_df_fixed.empty:
                mols2grid_html = mols2grid.display(
                    example_df_fixed,
                    smiles_col="SMILES",
                    n_cols=4,
                    template='static',
                    prerender=True,
                    size=(200, 200),
                    subset=["img", "ChEMBL_ID", "pIC50"],
                    transform={"pIC50": lambda x: f"{x:.2f}"},
                    selection_type=None,
                    _repr_html_=False 
                )
                st.components.v1.html(mols2grid_html, height=800, scrolling=True)
            else:
                st.info("No example compounds found for this transformation.")
    else:
        st.info("No transformations available for selection.")

if __name__ == "__main__":
    main()

# Save the Streamlit app to Google Drive
app_code = """
import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import Draw # For ReactionToImage
import io
import base64
from operator import itemgetter
from itertools import combinations
import mols2grid
import seaborn as sns
import matplotlib.pyplot as plt
import subprocess
import sys

# --- Install necessary packages ---
# This block ensures all required libraries are installed when the script runs.
# 'rdkit-pypi' is used for pip installation of RDKit.
def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        st.success(f"Successfully installed {package}")
    except Exception as e:
        st.error(f"Failed to install {package}: {e}. Please install manually if the app fails to load.")

install_package("streamlit")
install_package("pandas")
install_package("rdkit-pypi") 
install_package("mols2grid")
install_package("seaborn")
install_package("matplotlib")


# --- Custom functions from previous notebook steps ---

# From useful_rdkit_utils logic
def get_largest_fragment(mol):
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True)
    if not frags:
        return None
    num_atoms = [x.GetNumHeavyAtoms() for x in frags]
    idx = num_atoms.index(max(num_atoms))
    return frags[idx]

# Helper for sorting fragments and removing map numbers
def remove_map_nums(mol):
    if mol is None: return
    for atm in mol.GetAtoms():
        atm.SetAtomMapNum(0)

def sort_fragments(mol):
    if mol is None: return []
    frag_list = list(Chem.GetMolFrags(mol, asMols=True))
    [remove_map_nums(x) for x in frag_list]
    frag_num_atoms_list = [(x.GetNumAtoms(), x) for x in frag_list]
    frag_num_atoms_list.sort(key=itemgetter(0), reverse=True)
    return [x[1] for x in frag_num_atoms_list]

# From scaffold_finder.py
def GetRingSystems(mol, includeSpiro=True):
    ri = mol.GetRingInfo()
    systems = []
    for ring in ri.AtomRings():
        ringAts = set(ring)
        sys = set(ringAts)
        for otherRing in ri.AtomRings():
            if ringAts!=set(otherRing) and len(ringAts.intersection(otherRing))>0:
                sys = sys.union(otherRing)
        systems.append(tuple(sorted(list(sys))))
    systems.sort(key=lambda x:len(x),reverse=True)
    res = []
    seen = [False]*len(systems)
    for i,sysI in enumerate(systems):
        if seen[i]:
            continue
        res.append(sysI)
        for j in range(i+1,len(systems)):
            if not seen[j] and len(set(sysI).intersection(set(systems[j])))>0:
                seen[j]=True
    return res

def FragmentMol(mol,maxCuts=3,minMolSize=1,linker_smarts="[D3-D4;!R]"):
    # remove stereochemistry for fragmentation
    clean_mol = Chem.Mol(mol)
    Chem.RemoveStereochemistry(clean_mol)
    clean_mol = Chem.AddHs(clean_mol)
    
    # if maxCuts is 0, just return the original molecule
    if maxCuts == 0:
        return [(0,clean_mol)]

    res = []
    # find all possible linker cut sites
    patt = Chem.MolFromSmarts(linker_smarts)
    matches = clean_mol.GetSubstructMatches(patt)
    cut_bonds = set()
    for m in matches:
        for atm_idx in m:
            atm = clean_mol.GetAtomWithIdx(atm_idx)
            for bond in atm.GetBonds():
                if not bond.IsInRing(): # exclude cutting bonds in a ring
                    cut_bonds.add(tuple(sorted((bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()))))

    if not cut_bonds:
        return [(0,clean_mol)]

    # try all combinations of cuts up to maxCuts
    for num_cuts in range(1,maxCuts+1):
        if num_cuts > len(cut_bonds):
            continue
        for combo_bonds in combinations(cut_bonds,num_cuts):
            bonds_to_cut = []
            for b_atm1, b_atm2 in combo_bonds:
                bond = clean_mol.GetBondBetweenAtoms(b_atm1, b_atm2)
                if bond:
                    bonds_to_cut.append(bond.GetIdx())
            
            if bonds_to_cut:
                tmp = Chem.FragmentOnBonds(clean_mol,bonds_to_cut,addDummies=True)
                res.append((num_cuts,tmp))
    return res

# --- Image generation helper functions ---
def rxn_to_base64_image(rxn):
    if rxn is None: return ""
    try:
        # Use RDKit's Draw.ReactionToImage which returns a PIL Image
        img = Draw.ReactionToImage(rxn)
        img_buffer = io.BytesIO()
        img.save(img_buffer, format="PNG")
        im_text64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
        img_str = f"<img src='data:image/png;base64,{im_text64}' style='width: 300px; height: 150px;'/>"
        return img_str
    except Exception as e:
        return f"Error rendering reaction: {e}"

def stripplot_base64_image(dist):
    sns.set(rc={'figure.figsize': (3, 1)})
    sns.set_style('whitegrid')
    fig, ax = plt.subplots()
    sns.stripplot(x=dist, ax=ax)
    ax.axvline(0,ls="--",c="red")
    ax.set_xlim(-5,5)
    ax.set_yticks([]) 
    ax.set_xlabel("")
    s = io.BytesIO()
    plt.savefig(s, format='png', bbox_inches="tight")
    plt.close(fig)
    s = base64.b64encode(s.getvalue()).decode("utf-8").replace("\n", "")
    return '<img align="left" src="data:image/png;base64,%s">' % s

def find_examples(delta_df, query_idx):
    example_list = []
    if query_idx is None:
        return pd.DataFrame(columns=["SMILES","Name","pIC50"])
    
    if 'idx' not in delta_df.columns:
        st.error("Error: 'idx' column not found in delta_df. Cannot find examples.")
        return pd.DataFrame(columns=["SMILES","Name","pIC50"])

    if not delta_df.empty and query_idx in delta_df['idx'].unique():
        for idx,row in delta_df.query("idx == @query_idx").sort_values("Delta",ascending=False).iterrows():
            smi_1, name_1, pIC50_1 = row.SMILES_1, row.Name_1, row.pIC50_1
            smi_2, name_2, pIC50_2 = row.SMILES_2, row.Name_2, row.pIC50_2
            tmp_list = [(smi_1, name_1, pIC50_1),(smi_2, name_2, pIC50_2)]
            tmp_list.sort(key=itemgetter(0)) 
            example_list.append(tmp_list[0])
            example_list.append(tmp_list[1])
    return pd.DataFrame(example_list,columns=["SMILES","Name","pIC50"])

# --- Streamlit App ---
def main():
    st.set_page_config(layout="wide")
    st.title("Matched Molecular Pairs (MMPs) Analysis")

    st.markdown("""
        This application performs Matched Molecular Pairs (MMPs) analysis on a dataset of chemical compounds.
        It identifies pairs of molecules that differ by a small chemical transformation and analyzes the
        impact of these transformations on a property (e.g., pIC50).
    """)

    # 1. Load Data
    st.header("1. Load Data")
    uploaded_file = st.file_uploader("Upload your CSV file (e.g., MMP_9_MMP.csv)", type="csv")
    
    df = None
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("Data loaded successfully!")
        st.dataframe(df.head())
    else:
        st.info("Please upload a CSV file to begin.")
        # Attempt to load from default path if in Colab context (for testing/convenience)
        if 'google.colab' in sys.modules:
            default_path = "/content/drive/MyDrive/MMP_9_MMP.csv"
            try:
                df = pd.read_csv(default_path)
                st.info(f"Using default file from Google Drive: `{default_path}`")
                st.dataframe(df.head())
            except FileNotFoundError:
                st.warning(f"Default file `{default_path}` not found in your Google Drive. Please upload manually or ensure it's in `/content/drive/MyDrive/`.")
            except Exception as e:
                st.error(f"Error loading default file from Drive: {e}")
                
    if df is None or df.empty:
        st.stop() # Stop execution if no data is loaded or data is empty

    if not all(col in df.columns for col in ['SMILES', 'Name', 'pIC50']):
        st.error("The CSV file must contain 'SMILES', 'Name', and 'pIC50' columns.")
        st.stop()

    # 2. Preprocessing
    st.header("2. Preprocessing Molecules")
    
    # Create RDKit Mol objects
    st.write("Creating RDKit molecule objects...")
    mols = []
    mol_from_smiles_bar = st.progress(0, text="Converting SMILES to Mol objects...")
    for i, smiles in enumerate(df.SMILES):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                mols.append(mol)
            else:
                mols.append(None)
        except:
            mols.append(None)
        mol_from_smiles_bar.progress((i + 1) / len(df))
    df['mol'] = mols
    mol_from_smiles_bar.empty()

    # Filter out rows where mol creation failed
    df_valid_mols = df.dropna(subset=['mol']).copy()
    if df_valid_mols.empty:
        st.error("No valid molecules could be parsed from SMILES. Please check your data.")
        st.stop()
    
    # Get largest fragment
    st.write("Getting largest fragments...")
    largest_frags = []
    largest_frag_bar = st.progress(0, text="Extracting largest fragments...")
    for i, mol in enumerate(df_valid_mols['mol']):
        if mol:
            largest_frags.append(get_largest_fragment(mol))
        else:
            largest_frags.append(None)
        largest_frag_bar.progress((i + 1) / len(df_valid_mols))
    df_valid_mols.loc[:, 'mol'] = largest_frags # Use .loc for setting values
    largest_frag_bar.empty()
    st.success("Molecules preprocessed successfully.")

    # 3. Decompose Molecules to Scaffolds and Sidechains
    st.header("3. Decompose Molecules")
    
    row_list = []
    decompose_progress = st.progress(0, text="Decomposing molecules...")
    for i, row_data in enumerate(df_valid_mols[['SMILES', 'Name', 'pIC50', 'mol']].itertuples(index=False)):
        smiles, name, pIC50, mol = row_data
        decompose_progress.progress((i + 1) / len(df_valid_mols), text=f"Decomposing molecule {i+1}/{len(df_valid_mols)}")
        if mol is None: continue 
        
        frag_list = FragmentMol(mol, maxCuts=1) # maxCuts=1 as per notebook
        for _, frag_mol in frag_list:
            if frag_mol is None: continue 
            pair_list = sort_fragments(frag_mol)
            
            core_smiles = Chem.MolToSmiles(pair_list[0]) if len(pair_list) > 0 and pair_list[0] else None
            r_group_smiles = Chem.MolToSmiles(pair_list[1]) if len(pair_list) > 1 and pair_list[1] else None

            if core_smiles: 
                row_list.append([smiles, core_smiles, r_group_smiles, name, pIC50])
                
    decompose_progress.empty()
    row_df = pd.DataFrame(row_list, columns=["SMILES", "Core", "R_group", "Name", "pIC50"])
    st.subheader("Decomposed Molecules (first 10 rows)")
    st.dataframe(row_df.head(10))

    # 4. Collect Pairs With the Same Scaffold
    st.header("4. Collect Matched Molecular Pairs")

    delta_list = []
    collect_pairs_progress = st.progress(0, text="Collecting pairs...")
    
    grouped_cores = list(row_df.groupby("Core"))
    for i, (k, v) in enumerate(grouped_cores):
        collect_pairs_progress.progress((i + 1) / len(grouped_cores), text=f"Processing core {i+1}/{len(grouped_cores)}")
        if len(v) >= 2: 
            for a, b in combinations(range(0, len(v)), 2):
                reagent_a = v.iloc[a]
                reagent_b = v.iloc[b]
                if str(reagent_a.SMILES) == str(reagent_b.SMILES):
                    continue
                reagent_a, reagent_b = sorted([reagent_a, reagent_b], key=lambda x: str(x.SMILES))

                # Handle cases where R_group might be None (e.g., if only a core was found)
                if reagent_a.R_group is None or reagent_b.R_group is None:
                    continue
                
                delta = reagent_b.pIC50 - reagent_a.pIC50
                transform_smarts = f"{str(reagent_a.R_group).replace('*','*-')}>>{str(reagent_b.R_group).replace('*','*-')}"
                
                delta_list.append(list(reagent_a.values) + list(reagent_b.values) + [transform_smarts, delta])
    
    collect_pairs_progress.empty()
    cols = ["SMILES_1","Core_1","R_group_1","Name_1","pIC50_1",
            "SMILES_2","Core_2","R_group_2","Name_2","pIC50_2", 
            "Transform","Delta"]
    delta_df = pd.DataFrame(delta_list, columns=cols)
    st.subheader("Delta DataFrame (last 10 rows)")
    st.dataframe(delta_df.tail(10))

    # 5. Collect Frequently Occurring Pairs
    st.header("5. Summarize Transformations")
    min_transform_occurrence = st.slider("Minimum transform occurrence:", 1, 20, 5)

    mmp_list = []
    for k, v in delta_df.groupby("Transform"):
        if len(v) >= min_transform_occurrence:
            mmp_list.append([k, len(v), list(v.Delta.values)]) 
    
    mmp_df = pd.DataFrame(mmp_list, columns=["Transform", "Count", "Deltas"])
    
    if mmp_df.empty:
        st.warning("No frequent transformations found with the current minimum occurrence. Try lowering the value.")
        st.stop()

    mmp_df['idx'] = range(0, len(mmp_df))
    mmp_df['mean_delta'] = [x.mean() for x in mmp_df.Deltas]
    
    # Calculate rxn_mol and image representations
    st.info("Generating reaction images and distribution plots. This may take a moment...")
    rxn_mol_bar = st.progress(0, text="Generating reaction diagrams...")
    mmp_df['rxn_mol'] = None 
    for i, row in mmp_df.iterrows():
        try:
            mmp_df.loc[i, 'rxn_mol'] = AllChem.ReactionFromSmarts(row.Transform.replace('*-','*'), useSmiles=True)
        except Exception: # Catch specific RDKit errors if possible
            mmp_df.loc[i, 'rxn_mol'] = None 
        rxn_mol_bar.progress((i + 1) / len(mmp_df))
    rxn_mol_bar.empty()

    # Filter out rows where rxn_mol could not be created before applying image functions
    mmp_df = mmp_df.dropna(subset=['rxn_mol']).copy()
    if mmp_df.empty:
        st.warning("No valid transformations could be converted to RDKit reaction objects. Cannot display summary.")
        st.stop()

    mmp_df['MMP Transform'] = mmp_df.rxn_mol.apply(rxn_to_base64_image)
    mmp_df['Delta Distribution'] = mmp_df.Deltas.apply(stripplot_base64_image)

    st.subheader("Frequent Transformations Summary")
    
    sort_by_column = st.selectbox(
        "Sort transformations by:",
        options=["mean_delta", "Count"],
        index=0
    )
    ascending_order = st.checkbox("Sort ascending", value=False if sort_by_column == "mean_delta" else True)
    
    mmp_df_display = mmp_df.sort_values(sort_by_column, ascending=ascending_order).copy() # Use copy() to avoid SettingWithCopyWarning
    
    cols_to_show = ['MMP Transform','Count',"mean_delta","Delta Distribution"]
    # Adjust column names for the HTML table
    html_df = mmp_df_display[cols_to_show].round(2)
    html_df.columns = ['MMP Transform', 'Count', 'Mean ΔpIC50', 'Delta Distribution']

    st.markdown(
        html_df.to_html(escape=False),
        unsafe_allow_html=True
    )
    
    # 6. Explore Specific Transformation Examples
    st.header("6. Explore Specific Transformation Examples")

    if not mmp_df_display.empty:
        selected_transform_idx = st.selectbox(
            "Select a transformation to view examples:",
            options=mmp_df_display['idx'].tolist(),
            format_func=lambda x: f"{mmp_df_display.loc[mmp_df_display['idx'] == x, 'Transform'].iloc[0]} (Mean Δ: {mmp_df_display.loc[mmp_df_display['idx'] == x, 'mean_delta'].iloc[0]:.2f})"
        )

        if selected_transform_idx is not None:
            selected_transform_row = mmp_df.loc[mmp_df['idx'] == selected_transform_idx].iloc[0]
            st.subheader(f"Details for Transformation: {selected_transform_row['Transform']}")
            st.markdown(f"**Mean ΔpIC50:** {selected_transform_row['mean_delta']:.2f}")
            st.markdown(f"**Occurrences:** {selected_transform_row['Count']}")
            
            st.markdown("---")
            st.write("Reaction Diagram:")
            st.markdown(selected_transform_row['MMP Transform'], unsafe_allow_html=True)
            
            st.write("ΔpIC50 Distribution:")
            st.markdown(selected_transform_row['Delta Distribution'], unsafe_allow_html=True)

            st.subheader("Example Compound Pairs:")
            
            example_df = find_examples(delta_df, selected_transform_idx)
            example_df_fixed = example_df.rename(columns={"SMILES": "ChEMBL_ID", "Name": "SMILES"})

            if not example_df_fixed.empty:
                mols2grid_html = mols2grid.display(
                    example_df_fixed,
                    smiles_col="SMILES",
                    n_cols=4,
                    template='static',
                    prerender=True,
                    size=(200, 200),
                    subset=["img", "ChEMBL_ID", "pIC50"],
                    transform={"pIC50": lambda x: f"{x:.2f}"},
                    selection_type=None,
                    _repr_html_=False 
                )
                st.components.v1.html(mols2grid_html, height=800, scrolling=True)
            else:
                st.info("No example compounds found for this transformation.")
    else:
        st.info("No transformations available for selection.")

if __name__ == "__main__":
    main()
"""

with open("/content/drive/MyDrive/app.py", "w") as f:
    f.write(app_code)

print("`app.py` has been successfully saved to `/content/drive/MyDrive/app.py`.")
print("\nTo run the Streamlit application:")
print("1. Ensure your Google Drive is mounted (`from google.colab import drive; drive.mount('/content/drive')`).")
print("2. Make sure the `MMP_9_MMP.csv` file is located in `/content/drive/MyDrive/`.")
print("3. In a new cell, execute the following commands:")
print("   `!npm install localtunnel`")
print("   `!streamlit run /content/drive/MyDrive/app.py &>/content/logs.txt &`")
print("   `!npx localtunnel --port 8501`")
print("\nFollow the link provided by `localtunnel` to access your Streamlit app in your browser.")
