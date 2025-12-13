!pip install --quiet streamlit numpy seaborn matplotlib

import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from operator import itemgetter
from itertools import combinations
import mols2grid
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit.Chem.Draw import rdMolDraw2D
import io
import base64
import numpy as np # Explicitly import numpy for stripplot_base64_image

# Suppress warnings from RDKit
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# --- Embedded scaffold_finder.py content (relevant part) ---
# Source: https://raw.githubusercontent.com/PatWalters/practical_cheminformatics_tutorials/main/sar_analysis/scaffold_finder.py
def FragmentMol(mol,
                minCuts=1,
                maxCuts=3,
                minFragmentSize=1,
                maxFragmentSize=1000,
                bondSmarts="[!$([#0,#1,#5,#12,#13,#17,#20,#21,#22,#23,#24,#25,#26,#27,#28,#29,#30,#31,#32,#33,#34,#35,#39,#40,#41,#42,#43,#44,#45,#46,#47,#48,#49,#50,#51,#52,#53,#54,#55,#56,#57,#58,#59,#60,#61,#62,#63,#64,#65,#66,#67,#68,#69,#70,#71,#72,#73,#74,#75,#76,#77,#78,#79,#80,#81,#82,#83,#84,#85,#86,#87,#88,#89,#90,#91,#92,#93,#94,#95,#96,#97,#98,#99,#100,#101,#102,#103,#104,#105,#106,#107,#108,#109,#110,#111,#112,#113,#114,#115,#116,#117,#118)]-!@[!$([#0,#1,#5,#12,#13,#17,#20,#21,#22,#23,#24,#25,#26,#27,#28,#29,#30,#31,#32,#33,#34,#35,#39,#40,#41,#42,#43,#44,#45,#46,#47,#48,#49,#50,#51,#52,#53,#54,#55,#56,#57,#58,#59,#60,#61,#62,#63,#64,#65,#66,#67,#68,#69,#70,#71,#72,#73,#74,#75,#76,#77,#78,#79,#80,#81,#82,#83,#84,#85,#86,#87,#88,#89,#90,#91,#92,#93,#94,#95,#96,#97,#98,#99,#100,#101,#102,#103,#104,#105,#106,#107,#108,#109,#110,#111,#112,#113,#114,#115,#116,#117,#118])]",
                resultsAsMols=True,
                payload=None):
    if mol is None: return []
    res = []
    bonds = mol.GetSubstructMatches(Chem.MolFromSmarts(bondSmarts))
    if not bonds and payload is not None:
        return [(mol, payload)]
    elif not bonds:
        return [(mol,None)]

    bondIdxList = [x[0] for x in bonds]
    bondIdxList.sort()

    for numCuts in range(minCuts,maxCuts + 1):
        for tpl in combinations(bondIdxList,numCuts):
            tmpMol = Chem.Mol(mol)
            for idx in tpl:
                bond = tmpMol.GetBondWithIdx(idx)
                beginAtomIdx = bond.GetBeginAtomIdx()
                endAtomIdx = bond.GetEndAtomIdx()
                tmpMol = Chem.FragmentOnBonds(tmpMol, (idx,),
                                              addDummies=True,
                                              dummyLabels=[(beginAtomIdx,beginAtomIdx),
                                                           (endAtomIdx,endAtomIdx)])
            for f in Chem.GetMolFrags(tmpMol,asMols=True):
                is_fragment = False
                for atm in f.GetAtoms():
                    if atm.GetAtomicNum() == 0 and atm.GetIsotope() > 0 :
                        is_fragment = True
                        break
                if is_fragment:
                    if f.GetNumHeavyAtoms() >= minFragmentSize and \
                       f.GetNumHeavyAtoms() <= maxFragmentSize :
                        s = Chem.MolToSmiles(f,
                                             isomericSmiles=False,
                                             canonical=True)
                        if (s,s) not in res:
                            res.append((s,f))
    return res

# --- Embedded useful_rdkit_utils.py content (relevant part) ---
# Source: https://github.com/PatWalters/useful_rdkit_utils/blob/main/useful_rdkit_utils.py
def get_largest_fragment(mol):
    """
    Returns the largest fragment of a molecule.
    """
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True)
    if len(frags) == 0:
        return None
    elif len(frags) == 1:
        return mol
    else:
        return max(frags, key=lambda x: x.GetNumAtoms())

# --- Other helper functions from the notebook ---
def remove_map_nums(mol):
    """
    Remove atom map numbers from a molecule
    """
    for atm in mol.GetAtoms():
        atm.SetAtomMapNum(0)

def sort_fragments(mol):
    """
    Transform a molecule with multiple fragments into a list of molecules that is sorted by number of atoms
    from largest to smallest
    """
    frag_list = list(Chem.GetMolFrags(mol, asMols=True))
    [remove_map_nums(x) for x in frag_list]
    frag_num_atoms_list = [(x.GetNumAtoms(), x) for x in frag_list]
    frag_num_atoms_list.sort(key=itemgetter(0), reverse=True)
    return [x[1] for x in frag_num_atoms_list]

def rxn_to_base64_image(rxn):
    """
    Convert an RDKit reaction to an image
    """
    if rxn is None:
        return ""
    try:
        drawer = rdMolDraw2D.MolDraw2DCairo(300, 150)
        drawer.DrawReaction(rxn)
        drawer.FinishDrawing()
        img_bytes = drawer.GetDrawingText() # For Cairo, GetDrawingText() returns bytes
        im_text64 = base64.b64encode(img_bytes).decode('utf8')
        img_str = f"<img src='data:image/png;base64, {im_text64}' style='max-width: 100%; height: auto;'/>"
        return img_str
    except Exception as e:
        st.warning(f"Could not draw reaction: {e}")
        return "Failed to draw reaction"

def stripplot_base64_image(dist):
    """
    Plot a distribution as a seaborn stripplot and save the resulting image as a base64 image.
    """
    if not isinstance(dist, (list, tuple, pd.Series, np.ndarray)) or len(dist) == 0:
        return ""
    
    sns.set(rc={'figure.figsize': (3, 1)})
    sns.set_style('whitegrid')
    fig, ax = plt.subplots()
    sns.stripplot(x=dist, ax=ax, jitter=0.2) # Added jitter for better visualization
    ax.axvline(0, ls="--", c="red")
    
    # Adjust xlim dynamically, ensure there's a range even if all values are the same
    min_val = min(dist)
    max_val = max(dist)
    if min_val == max_val: # Handle cases where all deltas are identical
        ax.set_xlim(min_val - 1, max_val + 1)
    else:
        ax.set_xlim(min_val - (max_val - min_val) * 0.1, max_val + (max_val - min_val) * 0.1) # Add 10% padding
    
    ax.set_yticks([]) # Hide y-axis
    ax.set_xlabel("") # Hide x-axis label
    plt.tight_layout()
    
    s = io.BytesIO()
    plt.savefig(s, format='png', bbox_inches="tight")
    plt.close(fig) # Close the figure to prevent display issues
    s = base64.b64encode(s.getvalue()).decode("utf-8").replace("\n", "")
    return '<img align="left" src="data:image/png;base64,%s" style="max-width: 100%; height: auto;">' % s

def find_examples(delta_df, query_idx):
    """
    Finds example compound pairs for a given transformation index.
    Assumes delta_df.SMILES_1 and delta_df.SMILES_2 are actual SMILES strings,
    and delta_df.Name_1 and delta_df.Name_2 are ChEMBL IDs.
    """
    example_list = []
    if query_idx is None or query_idx not in delta_df['idx'].unique():
        return pd.DataFrame(columns=["SMILES","ChEMBL_ID","pIC50"])

    # Limit to a reasonable number of examples to prevent UI overload
    compound_pairs_for_idx = delta_df.query("idx == @query_idx").sort_values("Delta",ascending=False).head(10)

    for idx,row in compound_pairs_for_idx.iterrows():
        smi_1, name_1, pIC50_1 = row.SMILES_1, row.Name_1, row.pIC50_1
        smi_2, name_2, pIC50_2 = row.SMILES_2, row.Name_2, row.pIC50_2

        tmp_list = [(smi_1, name_1, pIC50_1),(smi_2, name_2, pIC50_2)]
        tmp_list.sort(key=itemgetter(0)) # Sort by SMILES string
        example_list.append(tmp_list[0])
        example_list.append(tmp_list[1])
    example_df = pd.DataFrame(example_list,columns=["SMILES","ChEMBL_ID","pIC50"])
    return example_df

# --- Streamlit App ---
def main():
    st.set_page_config(layout="wide")
    st.title("Matched Molecular Pairs (MMP) Analysis")

    st.markdown("""
    This application performs Matched Molecular Pairs (MMP) analysis to identify structural transformations
    that lead to significant changes in `pIC50`.

    **Instructions:**
    1.  Ensure you have `MMP_9_MMP.csv` in the same directory as this `app.py` file.
    2.  Run the application using `streamlit run app.py` in your terminal.
    """)

    # Load data
    data_path = "MMP_9_MMP.csv"
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        st.error(f"Error: Could not find '{data_path}'. "
                 "Please ensure `MMP_9_MMP.csv` is in the same directory as this app.py file, "
                 "or update the `data_path` variable in the code if it's located elsewhere.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    st.subheader("Original Data Sample")
    st.dataframe(df.head())

    st.subheader("Performing MMP Analysis...")
    
    # Using st.spinner for long running operations
    with st.spinner("Step 1/4: Processing molecules..."):
        df['mol'] = df.SMILES.apply(Chem.MolFromSmiles)
        df.mol = df.mol.apply(get_largest_fragment)
        df = df.dropna(subset=['mol']) # Remove rows where mol object couldn't be created

    with st.spinner("Step 2/4: Decomposing molecules to scaffolds and sidechains..."):
        row_list = []
        for smiles, name, pIC50, mol in df.values:
            frag_list = FragmentMol(mol, maxCuts=1)
            for _, frag_mol in frag_list:
                pair_list = sort_fragments(frag_mol)
                core_smiles = Chem.MolToSmiles(pair_list[0]) if len(pair_list) > 0 else None
                r_group_smiles = Chem.MolToSmiles(pair_list[1]) if len(pair_list) > 1 else None
                if core_smiles and r_group_smiles:
                    row_list.append([smiles, core_smiles, r_group_smiles, name, pIC50])
        row_df = pd.DataFrame(row_list, columns=["SMILES","Core","R_group","Name","pIC50"])
        row_df = row_df.dropna(subset=["Core", "R_group"])
        row_df = row_df[row_df["R_group"].apply(lambda x: isinstance(x, str))]

    with st.spinner("Step 3/4: Collecting pairs and calculating deltas..."):
        delta_list = []
        min_transform_occurrence_threshold = 5 # As defined in notebook
        
        # Using st.progress for visual feedback inside the spinner
        progress_bar_delta = st.progress(0)
        total_groups = len(row_df.groupby("Core"))
        processed_groups = 0

        for k, v in row_df.groupby("Core"):
            if len(v) >= 2: # At least two molecules to form a pair
                for a,b in combinations(range(0,len(v)),2):
                    reagent_a = v.iloc[a]
                    reagent_b = v.iloc[b]
                    if reagent_a.SMILES == reagent_b.SMILES:
                        continue
                    reagent_a, reagent_b = sorted([reagent_a, reagent_b], key=lambda x: x.SMILES)
                    delta = reagent_b.pIC50 - reagent_a.pIC50
                    delta_list.append([reagent_a.SMILES, reagent_a.Core, reagent_a.R_group, reagent_a.Name, reagent_a.pIC50,
                                       reagent_b.SMILES, reagent_b.Core, reagent_b.R_group, reagent_b.Name, reagent_b.pIC50,
                                       f"{reagent_a.R_group.replace('*','*-')}>>{reagent_b.R_group.replace('*','*-')}", delta])
            processed_groups += 1
            if total_groups > 0: # Avoid division by zero
                progress_bar_delta.progress(processed_groups / total_groups)
        progress_bar_delta.empty()

        cols = ["SMILES_1","Core_1","R_group_1","Name_1","pIC50_1",
                "SMILES_2","Core_2","R_group_2","Name_2","pIC50_2",
                "Transform","Delta"]
        delta_df = pd.DataFrame(delta_list,columns=cols)

        mmp_list = []
        for k,v in delta_df.groupby("Transform"):
            if len(v) > min_transform_occurrence_threshold:
                mmp_list.append([k, len(v), v.Delta.tolist()])
        mmp_df = pd.DataFrame(mmp_list,columns=["Transform","Count","Deltas"])
        mmp_df['idx'] = range(0,len(mmp_df))
        mmp_df['mean_delta'] = [sum(x)/len(x) if len(x)>0 else 0 for x in mmp_df.Deltas]
        mmp_df['rxn_mol'] = mmp_df.Transform.apply(lambda x: AllChem.ReactionFromSmarts(x.replace('*-','*'), useSmiles=True) if x is not None else None)

        transform_dict = dict([(a,b) for a,b in mmp_df[["Transform","idx"]].values])
        delta_df['idx'] = [transform_dict.get(x) for x in delta_df.Transform]
        delta_df = delta_df.dropna(subset=['idx'])
        delta_df['idx'] = delta_df['idx'].astype(int)

    with st.spinner("Step 4/4: Generating visualizations for MMP transforms (this may take a moment)..."):
        mmp_df['MMP Transform Image'] = mmp_df.rxn_mol.apply(rxn_to_base64_image)
        mmp_df['Delta Distribution Image'] = mmp_df.Deltas.apply(stripplot_base64_image)

    st.success("MMP Analysis Complete!")
    st.header("Matched Molecular Pairs Results")

    sort_order = st.radio("Sort transformations by:", ["Mean \u0394pIC50 (Descending)", "Mean \u0394pIC50 (Ascending)", "Count (Descending)", "Count (Ascending)"], index=0, horizontal=True)
    
    if "Mean \u0394pIC50 (Descending)" == sort_order:
        sorted_mmp_df = mmp_df.sort_values("mean_delta", ascending=False)
    elif "Mean \u0394pIC50 (Ascending)" == sort_order:
        sorted_mmp_df = mmp_df.sort_values("mean_delta", ascending=True)
    elif "Count (Descending)" == sort_order:
        sorted_mmp_df = mmp_df.sort_values("Count", ascending=False)
    else: # Count (Ascending)
        sorted_mmp_df = mmp_df.sort_values("Count", ascending=True)


    st.markdown("### Overview of Common Transformations")
    st.markdown(f"Displaying {len(sorted_mmp_df)} transformations that occurred more than {min_transform_occurrence_threshold} times.")
    
    # Store selected index in session state to maintain selection across reruns
    if 'selected_transform_idx' not in st.session_state:
        st.session_state.selected_transform_idx = None
    
    transform_options = sorted_mmp_df.apply(lambda x: f"{x['Transform']} (Count: {x['Count']}, Mean \u0394: {x['mean_delta']:.2f})", axis=1).tolist()
    
    # Pre-select the first option if no previous selection
    initial_index = 0
    if st.session_state.selected_transform_idx is not None:
        # Try to find the index of the previously selected transform
        try:
            prev_transform_text = mmp_df.loc[mmp_df['idx'] == st.session_state.selected_transform_idx].apply(
                lambda x: f"{x['Transform']} (Count: {x['Count']}, Mean \u0394: {x['mean_delta']:.2f})", axis=1
            ).iloc[0]
            initial_index = transform_options.index(prev_transform_text)
        except (IndexError, ValueError):
            initial_index = 0 # Fallback if not found

    selected_transform_text = st.selectbox(
        "Select a transformation for detailed view:",
        options=transform_options,
        index=initial_index
    )

    selected_idx = None
    if selected_transform_text:
        # Find the original row in mmp_df corresponding to the selected option
        # This requires iterating or using a dictionary lookup
        for index, row in sorted_mmp_df.iterrows():
            if f"{row['Transform']} (Count: {row['Count']}, Mean \u0394: {row['mean_delta']:.2f})" == selected_transform_text:
                selected_idx = row['idx']
                break
    
    if selected_idx is not None:
        st.session_state.selected_transform_idx = selected_idx
        
    if st.session_state.selected_transform_idx is not None:
        st.markdown(f"---")
        current_transform_row = mmp_df.loc[mmp_df['idx'] == st.session_state.selected_transform_idx].iloc[0]
        
        st.subheader(f"Details for Transformation: `{current_transform_row['Transform']}`")

        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("#### Reaction:")
            st.markdown(current_transform_row['MMP Transform Image'], unsafe_allow_html=True)
        with col2:
            st.markdown("#### Summary:")
            st.write(f"**Mean \u0394pIC50:** {current_transform_row['mean_delta']:.2f}")
            st.write(f"**Number of occurrences:** {current_transform_row['Count']}")
            st.markdown("#### \u0394pIC50 Distribution:")
            st.markdown(current_transform_row['Delta Distribution Image'], unsafe_allow_html=True)

        st.markdown("#### Example Compound Pairs:")
        example_df_for_display = find_examples(delta_df, st.session_state.selected_transform_idx)
        
        if not example_df_for_display.empty:
            m2g_html = mols2grid.display(
                example_df_for_display,
                smiles_col="SMILES",
                n_cols=4,
                template='static',
                prerender=True,
                size=(200, 200),
                subset=["img", "ChEMBL_ID", "pIC50"],
                transform={"pIC50": lambda x: f"{x:.2f}"},
                tooltip=["SMILES", "ChEMBL_ID", "pIC50"]
            )._repr_html_()
            st.components.v1.html(m2g_html, height=500, scrolling=True)
        else:
            st.info("No examples found for this transformation.")

    st.markdown("---")
    st.markdown("Developed using RDKit, Pandas, Seaborn, Matplotlib, mols2grid, and Streamlit.")

if __name__ == '__main__':
    main()
