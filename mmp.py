import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from operator import itemgetter
from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
import requests
import sys

# --- RDKit Utility Functions (Copied/Adapted from the Notebook) ---
# Note: You would normally put these in a separate utility file (e.g., utils.py)
# For this self-contained app.py, they are included here.

# Download scaffold_finder.py from GitHub (required for FragmentMol)
@st.cache_resource
def get_scaffold_finder():
    """Downloads and imports the scaffold_finder library."""
    try:
        import scaffold_finder
        return scaffold_finder.FragmentMol
    except ImportError:
        # Download the file
        try:
            st.warning("Downloading 'scaffold_finder.py'. This only happens once.")
            lib_url = "https://raw.githubusercontent.com/PatWalters/practical_cheminformatics_tutorials/main/sar_analysis/scaffold_finder.py"
            lib_file = requests.get(lib_url)
            with open("scaffold_finder.py", "w") as ofs:
                ofs.write(lib_file.text)
            
            # Dynamically import the module
            import importlib.util
            spec = importlib.util.spec_from_file_location("scaffold_finder", "scaffold_finder.py")
            scaffold_finder = importlib.util.module_from_spec(spec)
            sys.modules["scaffold_finder"] = scaffold_finder
            spec.loader.exec_module(scaffold_finder)
            return scaffold_finder.FragmentMol
        except Exception as e:
            st.error(f"Could not download or import scaffold_finder: {e}")
            return None


@st.cache_resource
def get_useful_rdkit_utils():
    """Dynamically imports get_largest_fragment logic."""
    # Simplified version of get_largest_fragment to avoid installing another external library
    def get_largest_fragment(mol):
        if not mol:
            return None
        frags = Chem.GetMolFrags(mol, asMols=True)
        if not frags:
            return None
        largest_frag = max(frags, key=lambda m: m.GetNumAtoms())
        return largest_frag
    return get_largest_fragment

# Initialize external functions
FragmentMol = get_scaffold_finder()
get_largest_fragment = get_useful_rdkit_utils()


def remove_map_nums(mol):
    """Remove atom map numbers from a molecule"""
    if not mol: return
    for atm in mol.GetAtoms():
        atm.SetAtomMapNum(0)

def sort_fragments(mol):
    """
    Transform a molecule with multiple fragments into a list of molecules that is sorted 
    by number of atoms from largest to smallest
    """
    if not mol: return []
    frag_list = list(Chem.GetMolFrags(mol, asMols=True))
    [remove_map_nums(x) for x in frag_list]
    frag_num_atoms_list = [(x.GetNumAtoms(), x) for x in frag_list]
    frag_num_atoms_list.sort(key=itemgetter(0), reverse=True)
    return [x[1] for x in frag_num_atoms_list]

def rxn_to_base64_image(rxn):
    """Convert an RDKit reaction to a base64 image string for HTML display."""
    try:
        drawer = rdMolDraw2D.MolDraw2DCairo(300, 150)
        # Set options for better visual
        opts = drawer.drawOptions()
        opts.padding = 0.05
        drawer.DrawReaction(rxn)
        drawer.FinishDrawing()
        
        # Get PNG bytes
        png_bytes = drawer.GetDrawingText()
        im_text64 = base64.b64encode(png_bytes).decode('utf8')
        return f"<img src='data:image/png;base64, {im_text64}' style='max-width: 100%; height: auto;'/>"
    except Exception:
        return "<span>Error rendering reaction</span>"

def strippplot_base64_image(dist):
    """Plot distribution as a seaborn stripplot and save as a base64 image."""
    try:
        fig, ax = plt.subplots(figsize=(3, 1))
        sns.stripplot(x=dist, ax=ax, color=sns.color_palette()[0])
        ax.axvline(0, ls="--", c="red")
        ax.set_xlim(-5, 5)
        ax.set_yticks([]) # Hide Y-axis labels
        ax.set_xlabel("$\Delta pIC_{50}$")
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        s = io.BytesIO()
        plt.savefig(s, format='png', bbox_inches="tight")
        plt.close(fig)
        s = base64.b64encode(s.getvalue()).decode("utf-8").replace("\n", "")
        return f'<img src="data:image/png;base64,{s}" style="max-width: 100%; height: auto;">'
    except Exception:
        plt.close() # Ensure figure is closed on error
        return "<span>Error rendering plot</span>"
        
def mol_to_base64_image(mol, size=(200, 200)):
    """Convert an RDKit molecule to a base64 image string for HTML display."""
    if not mol:
        return "<span>No Mol</span>"
    try:
        drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        png_bytes = drawer.GetDrawingText()
        im_text64 = base64.b64encode(png_bytes).decode('utf8')
        return f"<img src='data:image/png;base64, {im_text64}' style='width: {size[0]}px; height: {size[1]}px;'/>"
    except Exception:
        return "<span>Error rendering molecule</span>"

# --- Main MMP Logic Adapted for Streamlit Caching ---

@st.cache_data(show_spinner="Loading and Pre-processing Data...")
def load_and_process_data(url="https://raw.githubusercontent.com/PatWalters/practical_cheminformatics_tutorials/main/data/hERG.csv"):
    """Load data, create RDKit mols, and remove salts/counterions."""
    df = pd.read_csv(url)
    df['mol'] = df.SMILES.apply(Chem.MolFromSmiles)
    df.mol = df.mol.apply(get_largest_fragment)
    df.dropna(subset=['mol'], inplace=True)
    return df

@st.cache_data(show_spinner="Decomposing Molecules to Scaffolds and Sidechains...")
def decompose_molecules(df_input):
    """Convert molecules to scaffold/sidechain pairs (Core, R_group)."""
    if FragmentMol is None:
        return pd.DataFrame() # Return empty if dependencies failed
    
    row_list = []
    # Using iterrows for simplicity in Streamlit caching context, though less efficient than numpy vectorization
    for index, row in df_input.iterrows():
        smiles, name, pIC50, mol = row.SMILES, row.Name, row.pIC50, row.mol
        if mol is None: continue
        
        # FragmentMol is the core of the decomposition
        frag_list = FragmentMol(mol, maxCuts=1)
        
        for _, frag_mol in frag_list:
            pair_list = sort_fragments(frag_mol)
            # We expect exactly two fragments: Core and R_group. If not, skip.
            if len(pair_list) == 2:
                tmp_list = [smiles] + [Chem.MolToSmiles(x) for x in pair_list] + [name, pIC50]
                row_list.append(tmp_list)

    cols = ["SMILES", "Core", "R_group", "Name", "pIC50"]
    return pd.DataFrame(row_list, columns=cols)

@st.cache_data(show_spinner="Identifying Matched Molecular Pairs (MMPs)...")
def identify_mmp_pairs(row_df):
    """Find pairs sharing the same Core and calculate Delta pIC50."""
    delta_list = []
    
    # Group by Core and iterate over combinations
    for _, v in row_df.groupby("Core"):
        if len(v) >= 2:
            # Iterate over all unique pairs of indices in the group
            for a, b in combinations(range(0, len(v)), 2):
                reagent_a = v.iloc[a]
                reagent_b = v.iloc[b]
                
                if reagent_a.SMILES == reagent_b.SMILES:
                    continue
                
                # Canonical sorting based on SMILES
                reagent_a, reagent_b = sorted([reagent_a, reagent_b], key=lambda x: x.SMILES)
                
                delta = reagent_b.pIC50 - reagent_a.pIC50
                
                # Create the transformation string: R_group_1 >> R_group_2
                # The replace('*','*-') is to ensure '*' is visible for the RDKit reaction smarts
                transform_str = f"{reagent_a.R_group.replace('*','*-')}>>{reagent_b.R_group.replace('*','*-')}"
                
                delta_list.append(list(reagent_a.values) + list(reagent_b.values) + [transform_str, delta])

    cols = ["SMILES_1", "Core_1", "R_group_1", "Name_1", "pIC50_1",
            "SMILES_2", "Core_2", "R_group_2", "Name_2", "pIC50_2", # R_group_2 was Rgroup_1 in notebook
            "Transform", "Delta"]
    return pd.DataFrame(delta_list, columns=cols)

@st.cache_data(show_spinner="Summarizing Transforms...")
def summarize_mmp_transforms(delta_df, min_occurrence):
    """Summarize frequent transforms and calculate mean delta pIC50."""
    
    mmp_list = []
    for k, v in delta_df.groupby("Transform"):
        if len(v) >= min_occurrence:
            mmp_list.append([k, len(v), v.Delta.values])
            
    mmp_df = pd.DataFrame(mmp_list, columns=["Transform", "Count", "Deltas"])
    mmp_df['idx'] = range(0, len(mmp_df))
    mmp_df['mean_delta'] = [x.mean() for x in mmp_df.Deltas]
    
    # Create the RDKit reaction object (cached since it's an RDKit object)
    mmp_df['rxn_mol'] = mmp_df.Transform.apply(lambda x: AllChem.ReactionFromSmarts(x, useSmiles=True))
    
    # Prepare base64 images for HTML table display (expensive operation)
    mmp_df['MMP Transform'] = mmp_df.rxn_mol.apply(rxn_to_base64_image)
    mmp_df['Delta Distribution'] = mmp_df.Deltas.apply(strippplot_base64_image)
    
    # Link the two dataframes
    transform_dict = dict([(a, b) for a, b in mmp_df[["Transform", "idx"]].values])
    delta_df['idx'] = [transform_dict.get(x) for x in delta_df.Transform]
    
    return mmp_df, delta_df

def find_examples(delta_df, query_idx):
    """Extract all molecular pairs for a given transform index (query_idx)."""
    
    example_list = []
    
    # Filter and sort by Delta
    query_df = delta_df.query("idx == @query_idx").sort_values("Delta", ascending=False)
    
    for _, row in query_df.iterrows():
        # Pair 1
        smi_1, name_1, pIC50_1 = row.SMILES_1, row.Name_1, row.pIC50_1
        # Pair 2
        smi_2, name_2, pIC50_2 = row.SMILES_2, row.Name_2, row.pIC50_2
        
        # Sort the pairs based on SMILES (canonical ordering from before)
        tmp_list = [(smi_1, name_1, pIC50_1), (smi_2, name_2, pIC50_2)]
        tmp_list.sort(key=itemgetter(0))
        
        example_list.append(tmp_list[0])
        example_list.append(tmp_list[1])
        
    example_df = pd.DataFrame(example_list, columns=["SMILES", "Name", "pIC50"])
    
    # Create RDKit molecules and images for display
    example_df['mol'] = example_df.SMILES.apply(Chem.MolFromSmiles)
    example_df['Image'] = example_df.mol.apply(lambda x: mol_to_base64_image(x, size=(150, 150)))
    
    return example_df

# --- Streamlit App Layout ---

def main():
    st.set_page_config(layout="wide")
    st.title("Matched Molecular Pair (MMP) Analysis")
    st.markdown("Examining the impact of chemical changes on hERG inhibition using MMPs.")
    st.markdown("---")

    # 1. Load Data
    data_df = load_and_process_data()

    if data_df.empty:
        st.error("Data loading or initial processing failed. Check console for dependencies.")
        return

    st.subheader("1. Data Summary")
    st.write(f"Loaded {len(data_df)} unique molecules (after salt stripping).")
    st.dataframe(data_df[['SMILES', 'Name', 'pIC50']].head(), use_container_width=True)
    st.markdown("---")

    # 2. Decompose Molecules
    row_df = decompose_molecules(data_df)
    if row_df.empty:
        st.error("Molecular decomposition failed. Check if 'scaffold_finder' downloaded correctly.")
        return
        
    # 3. Identify MMP Pairs
    delta_df = identify_mmp_pairs(row_df)

    # 4. Summarize MMP Transforms (User Controls)
    st.subheader("2. MMP Transform Analysis")

    col1, col2 = st.columns(2)
    with col1:
        min_transform_occurrence = st.slider(
            "Minimum Transform Occurrence:",
            min_value=2, max_value=50, value=5, step=1,
            help="Minimum number of times a transformation must occur to be considered an MMP."
        )
    with col2:
        rows_to_show = st.slider(
            "Number of Top MMPs to Display:",
            min_value=5, max_value=50, value=10, step=5,
            help="Number of most impactful MMPs to show in the table."
        )
    
    mmp_df, delta_df = summarize_mmp_transforms(delta_df, min_transform_occurrence)

    # Display the results table
    if mmp_df.empty:
        st.warning(f"No MMPs found with minimum occurrence of {min_transform_occurrence}.")
        return

    st.markdown(f"Found **{len(mmp_df)}** unique MMP transforms that occur at least {min_transform_occurrence} times.")

    sort_order = st.radio(
        "Sort Order (Impact on hERG pIC50):",
        ["Decrease hERG Activity (Lower pIC50)", "Increase hERG Activity (Higher pIC50)"],
        index=1,
        help="Lower pIC50 means higher IC50 (reduced hERG activity - a desirable outcome)."
    )
    ascending = True if "Decrease" in sort_order else False

    mmp_df.sort_values("mean_delta", inplace=True, ascending=ascending)

    # Construct and display HTML table
    st.markdown("### Top Matched Molecular Pairs")
    html_table = mmp_df[['MMP Transform', 'Count', "mean_delta", "Delta Distribution"]]\
        .round({'mean_delta': 2})\
        .head(rows_to_show)\
        .to_html(escape=False, index=True, justify='left', classes='table table-striped')
    
    st.components.v1.html(
        f"""
        <style>
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9em;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: center;
            vertical-align: middle;
        }}
        th:first-child, td:first-child {{
            width: 50px; /* Index column */
        }}
        th:nth-child(2), td:nth-child(2) {{
            width: 350px; /* Reaction Image */
        }}
        th:nth-child(4), td:nth-child(4) {{
            width: 100px; /* Mean Delta */
        }}
        th:nth-child(5), td:nth-child(5) {{
            width: 350px; /* Plot Image */
        }}
        </style>
        {html_table}
        """,
        height=int(180 * rows_to_show) + 50, # Estimate height
        scrolling=True
    )
    
    # 5. Visualize Examples (User Interaction)
    st.markdown("---")
    st.subheader("3. Visualize Molecular Examples")

    # Get the index (the far left column in the HTML table)
    top_indices = mmp_df.head(rows_to_show).index.tolist()
    index_map = {f"Index {idx} | Mean $\Delta pIC_{{50}}$: {mmp_df.loc[idx, 'mean_delta']:.2f}": idx for idx in top_indices}
    
    # Ensure a default selection exists
    if not top_indices:
        st.warning("No MMPs to show examples for.")
        return

    selection_key = st.selectbox(
        "Select an MMP Index to View Examples:",
        options=list(index_map.keys()),
        help="Select the index from the far-left column of the table above to view all pairs contributing to that MMP."
    )
    query_idx = index_map[selection_key]
    
    st.markdown(f"**Viewing Examples for MMP Index: {query_idx}**")
    
    example_df = find_examples(delta_df, query_idx)
    
    # Display the examples in a clean, multi-column format (replacing mols2grid)
    example_cols = st.columns(4)
    for i, row in example_df.iterrows():
        col = example_cols[i % 4]
        with col:
            st.markdown(f"**Compound {i + 1}**")
            st.markdown(f"pIC50: **{row.pIC50:.2f}**")
            st.markdown(f"Name: *{row.Name}*")
            # Display image using HTML component
            st.components.v1.html(row.Image, height=180)


if __name__ == "__main__":
    main()
