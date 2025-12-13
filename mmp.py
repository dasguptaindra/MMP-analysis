import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdMMPA import FragmentMol
from rdkit.Chem.rdRGroupDecomposition import RGroupDecompose
from rdkit.Chem import Draw
import itertools
import sys
import base64
from io import BytesIO

# --- Helper Functions (From useful_rdkit_utils) ---
def get_largest_fragment(mol):
    """
    Standalone implementation of useful_rdkit_utils.get_largest_fragment
    """
    frags = Chem.GetMolFrags(mol, asMols=True)
    if not frags:
        return mol
    return max(frags, key=lambda x: x.GetNumAtoms())

# --- Core Logic (From your script) ---

def cleanup_fragment(mol):
    """
    Replace atom map numbers with Hydrogens
    :param mol: input molecule
    :return: modified molecule, number of R-groups
    """
    rgroup_count = 0
    for atm in mol.GetAtoms():
        atm.SetAtomMapNum(0)
        if atm.GetAtomicNum() == 0:
            rgroup_count += 1
            atm.SetAtomicNum(1)
    mol = Chem.RemoveAllHs(mol)
    return mol, rgroup_count

def generate_fragments(mol):
    """
    Generate fragments using the RDKit
    :param mol: RDKit molecule
    :return: a Pandas dataframe with Scaffold SMILES, Number of Atoms, Number of R-Groups
    """
    # Generate molecule fragments
    frag_list = FragmentMol(mol)
    # Flatten the output into a single list
    flat_frag_list = [x for x in itertools.chain(*frag_list) if x]
    # The output of Fragment mol is contained in single molecules.  Extract the largest fragment from each molecule
    flat_frag_list = [get_largest_fragment(x) for x in flat_frag_list]
    
    # Keep fragments where the number of atoms in the fragment is at least 2/3 of the number fragments in
    # input molecule
    num_mol_atoms = mol.GetNumAtoms()
    if num_mol_atoms == 0: return pd.DataFrame(columns=["Scaffold", "NumAtoms", "NumRgroupgs"])
    
    flat_frag_list = [x for x in flat_frag_list if x.GetNumAtoms() / num_mol_atoms > 0.67]
    # remove atom map numbers from the fragments
    flat_frag_list = [cleanup_fragment(x) for x in flat_frag_list]
    # Convert fragments to SMILES
    frag_smiles_list = [[Chem.MolToSmiles(x), x.GetNumAtoms(), y] for (x, y) in flat_frag_list]
    # Add the input molecule to the fragment list
    frag_smiles_list.append([Chem.MolToSmiles(mol), mol.GetNumAtoms(), 1])
    # Put the results into a Pandas dataframe
    frag_df = pd.DataFrame(frag_smiles_list, columns=["Scaffold", "NumAtoms", "NumRgroupgs"])
    # Remove duplicate fragments
    frag_df = frag_df.drop_duplicates("Scaffold")
    return frag_df

def find_scaffolds(df_in):
    """
    Generate scaffolds for a set of molecules
    :param df_in: Pandas dataframe with [SMILES, Name, RDKit molecule] columns
    :return: dataframe with molecules and scaffolds, dataframe with unique scaffolds
    """
    # Loop over molecules and generate fragments
    df_list = []
    
    # Using st.progress for feedback loop
    progress_bar = st.progress(0)
    total_mols = len(df_in)
    
    for i, (smiles, name, mol) in enumerate(df_in[["SMILES", "Name", "mol"]].values):
        if mol is None: continue
        tmp_df = generate_fragments(mol).copy()
        tmp_df['Name'] = name
        tmp_df['SMILES'] = smiles
        df_list.append(tmp_df)
        
        # Update progress every 5%
        if i % (max(1, total_mols // 20)) == 0:
            progress_bar.progress(min(i / total_mols, 1.0))
            
    progress_bar.progress(1.0)
    
    if not df_list:
        return pd.DataFrame(), pd.DataFrame()

    # Combine the list of dataframes into a single dataframe
    mol_df = pd.concat(df_list)
    
    # Collect scaffolds
    scaffold_list = []
    # GroupBy is faster than iterating if we just want aggregation, but keeping script logic structure
    group_stats = mol_df.groupby("Scaffold").agg(
        Count=('Name', 'nunique'),
        NumAtoms=('NumAtoms', 'first')
    ).reset_index()
    
    scaffold_df = group_stats.copy()
    
    # Any fragment that occurs more times than the number of fragments can't be a scaffold
    # FIX: Defined num_df_rows based on input df length
    num_df_rows = len(df_in)
    scaffold_df = scaffold_df.query("Count <= @num_df_rows")
    
    # Sort scaffolds by frequency
    scaffold_df = scaffold_df.sort_values(["Count", "NumAtoms"], ascending=[False, False])
    return mol_df, scaffold_df

def get_molecules_with_scaffold(scaffold, mol_df, activity_df):
    """
    Associate molecules with scaffolds
    :param scaffold: scaffold SMILES
    :param mol_df: dataframe with molecules and scaffolds
    :param activity_df: dataframe with [SMILES, Name, pIC50] columns
    :return: list of core(s), dataframe with [SMILES, Name, pIC50]
    """
    match_df = mol_df.query("Scaffold == @scaffold")
    merge_df = match_df.merge(activity_df, on=["SMILES", "Name"])
    scaffold_mol = Chem.MolFromSmiles(scaffold)
    
    if scaffold_mol is None:
        return [], merge_df[["SMILES", "Name", "pIC50"]]

    rgroup_match, rgroup_miss = RGroupDecompose(scaffold_mol, merge_df.mol, asSmiles=True)
    
    if len(rgroup_match):
        rgroup_df = pd.DataFrame(rgroup_match)
        # Handle case where 'Core' might not be in columns if decomposition failed oddly
        if 'Core' in rgroup_df.columns:
            return rgroup_df.Core.unique(), merge_df[["SMILES", "Name", "pIC50"]]
        else:
            return [], merge_df[["SMILES", "Name", "pIC50"]]
    else:
        return [], merge_df[["SMILES", "Name", "pIC50"]]

def mol_to_image_html(smiles):
    """Generate HTML image tag for a SMILES string"""
    if not smiles: return ""
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return "Invalid SMILES"
    img = Draw.MolToImage(mol, size=(200, 200))
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f'<img src="data:image/png;base64,{img_str}" width="200"/>'

# --- Streamlit App Structure ---

def main():
    st.set_page_config(page_title="MMP Scaffold Analysis", layout="wide")
    st.title("🧪 MMP Scaffold Analysis")
    st.markdown("""
    This application performs Matched Molecular Pair (MMP) analysis to identify common scaffolds 
    and analyze activity distributions (pIC50) for molecules sharing the same core.
    """)

    # Sidebar for Inputs
    st.sidebar.header("Data Upload")
    uploaded_file = st.sidebar.file_uploader("Upload SMILES/CSV File", type=["csv", "smi", "txt"])
    
    if uploaded_file:
        # Load Data
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else: # Assume whitespace separated
                df = pd.read_csv(uploaded_file, sep=r'\s+', engine='python', header=None)
                # Try to assign reasonable default names if headers are missing
                if len(df.columns) >= 3:
                     df.columns = ["SMILES", "Name", "pIC50"] + list(df.columns[3:])
                else:
                    st.error("Input file requires at least 3 columns: SMILES, Name, pIC50")
                    return
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return

        st.sidebar.subheader("Column Mapping")
        cols = df.columns.tolist()
        
        # Try to guess columns
        default_smi = next((c for c in cols if "smi" in c.lower()), cols[0])
        default_name = next((c for c in cols if "name" in c.lower() or "id" in c.lower()), cols[1] if len(cols)>1 else cols[0])
        default_act = next((c for c in cols if "pic50" in c.lower() or "act" in c.lower()), cols[2] if len(cols)>2 else cols[0])

        smi_col = st.sidebar.selectbox("SMILES Column", cols, index=cols.index(default_smi))
        name_col = st.sidebar.selectbox("ID/Name Column", cols, index=cols.index(default_name))
        act_col = st.sidebar.selectbox("Activity (pIC50) Column", cols, index=cols.index(default_act))

        # Standardize DF
        input_df = df[[smi_col, name_col, act_col]].copy()
        input_df.columns = ["SMILES", "Name", "pIC50"]
        
        st.write(f"**Loaded Data:** {len(input_df)} molecules")
        with st.expander("Preview Input Data"):
            st.dataframe(input_df.head())

        # Run Analysis
        if st.sidebar.button("Run Scaffold Analysis"):
            with st.spinner("Generating Fragments & Scaffolds..."):
                # Pre-calculate Mols
                input_df['mol'] = input_df.SMILES.apply(Chem.MolFromSmiles)
                # Filter invalid mols
                valid_mols = input_df.dropna(subset=['mol'])
                if len(valid_mols) < len(input_df):
                    st.warning(f"Removed {len(input_df) - len(valid_mols)} invalid SMILES.")
                
                # Run Logic
                mol_df, scaffold_df = find_scaffolds(valid_mols)
                
                # Save to session state to persist
                st.session_state['mol_df'] = mol_df
                st.session_state['scaffold_df'] = scaffold_df
                st.session_state['input_df'] = valid_mols
                st.session_state['has_run'] = True
                st.success("Analysis Complete!")

    # Display Results
    if st.session_state.get('has_run', False):
        scaffold_df = st.session_state['scaffold_df']
        mol_df = st.session_state['mol_df']
        input_df = st.session_state['input_df']

        st.divider()
        st.subheader("1. Identified Scaffolds")
        st.markdown("Select a scaffold below to view molecule details.")
        
        # Interactive Grid for Scaffolds
        # We can't put images easily in a dataframe, so we use a selectbox or a grid view logic
        # For simplicity and speed: Table of stats, select via ID
        
        # Filter options
        min_count = st.slider("Min Molecules per Scaffold", 1, int(scaffold_df['Count'].max()), 2)
        filtered_scaffolds = scaffold_df[scaffold_df['Count'] >= min_count]
        
        st.dataframe(filtered_scaffolds, use_container_width=True)
        
        # Selection
        selected_scaffold_smiles = st.selectbox(
            "Select a Scaffold to Analyze", 
            options=filtered_scaffolds['Scaffold'].values,
            format_func=lambda x: f"{x} (n={filtered_scaffolds[filtered_scaffolds.Scaffold==x].Count.values[0]})"
        )

        if selected_scaffold_smiles:
            st.divider()
            st.subheader("2. Scaffold Details")
            
            # Get data
            cores, matches_df = get_molecules_with_scaffold(
                selected_scaffold_smiles, mol_df, input_df
            )
            
            # Layout: Left = Structures, Right = Stats/Table
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("**Selected Scaffold**")
                st.image(Draw.MolToImage(Chem.MolFromSmiles(selected_scaffold_smiles)), caption="Scaffold Query")
                
                if len(cores) > 0:
                    st.markdown("**Identified Core (with R-labels)**")
                    # Usually only 1 core if RGroupDecompose works cleanly
                    for core_smi in cores:
                        st.image(Draw.MolToImage(Chem.MolFromSmiles(core_smi)), caption="Core w/ R-groups")
                else:
                    st.info("No R-Group Decomposition core found.")

            with col2:
                ic50_vals = matches_df['pIC50'].astype(float)
                stats = {
                    "Molecules Count": len(matches_df),
                    "pIC50 Range": f"{ic50_vals.min():.2f} - {ic50_vals.max():.2f}",
                    "pIC50 Delta": f"{ic50_vals.max() - ic50_vals.min():.2f}",
                    "pIC50 Std Dev": f"{ic50_vals.std():.3f}"
                }
                st.json(stats)
                
                st.markdown("### Molecules")
                st.dataframe(matches_df.sort_values("pIC50", ascending=False), use_container_width=True)

if __name__ == "__main__":
    main()
