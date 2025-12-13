import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdMMPA import FragmentMol
from rdkit.Chem.rdRGroupDecomposition import RGroupDecompose
from rdkit.Chem import Draw
import itertools
import base64
from io import BytesIO

# --- 1. Helper Functions (Logic from scaffold_finder.py) ---

def get_largest_fragment(mol):
    """
    Standalone implementation to extract largest fragment.
    """
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True)
    if not frags:
        return mol
    return max(frags, key=lambda x: x.GetNumAtoms())

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
    try:
        frag_list = FragmentMol(mol)
    except Exception:
        return pd.DataFrame(columns=["Scaffold", "NumAtoms", "NumRgroupgs"])

    # Flatten the output into a single list
    flat_frag_list = [x for x in itertools.chain(*frag_list) if x]
    
    # Extract the largest fragment from each molecule
    flat_frag_list = [get_largest_fragment(x) for x in flat_frag_list]
    
    # Keep fragments where the number of atoms in the fragment is at least 2/3 of input molecule
    num_mol_atoms = mol.GetNumAtoms()
    if num_mol_atoms == 0:
        return pd.DataFrame(columns=["Scaffold", "NumAtoms", "NumRgroupgs"])
        
    flat_frag_list = [x for x in flat_frag_list if x and (x.GetNumAtoms() / num_mol_atoms > 0.67)]
    
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
    df_list = []
    
    # Progress bar setup
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_mols = len(df_in)
    
    # Loop over molecules and generate fragments
    for i, (smiles, name, mol) in enumerate(df_in[["SMILES", "Name", "mol"]].values):
        if mol is None: 
            continue
            
        try:
            tmp_df = generate_fragments(mol).copy()
            tmp_df['Name'] = name
            tmp_df['SMILES'] = smiles
            df_list.append(tmp_df)
        except Exception:
            pass # Skip molecules that fail fragmentation
        
        # Update progress occasionally to save render time
        if i % 10 == 0 or i == total_mols - 1:
            progress = min((i + 1) / total_mols, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Processing molecule {i+1} of {total_mols}")

    status_text.empty()
    progress_bar.empty()

    if not df_list:
        return pd.DataFrame(), pd.DataFrame()

    # Combine the list of dataframes into a single dataframe
    mol_df = pd.concat(df_list)
    
    # Collect scaffolds statistics
    # Using groupby to match logic: count unique Names per scaffold
    scaffold_stats = mol_df.groupby("Scaffold").agg({
        "Name": "nunique",
        "NumAtoms": "first"
    }).reset_index()
    scaffold_stats.columns = ["Scaffold", "Count", "NumAtoms"]
    
    scaffold_df = scaffold_stats
    
    # Any fragment that occurs more times than the number of fragments can't be a scaffold
    # Logic note: if a scaffold is the full molecule for every molecule, count == len(df_in)
    # The original script uses <= num_df_rows, effectively keeping everything. 
    num_df_rows = len(df_in)
    scaffold_df = scaffold_df.query("Count <= @num_df_rows")
    
    # Sort scaffolds by frequency
    scaffold_df = scaffold_df.sort_values(["Count", "NumAtoms"], ascending=[False, False])
    
    return mol_df, scaffold_df

def get_molecules_with_scaffold(scaffold, mol_df, activity_df):
    """
    Associate molecules with scaffolds with robust column handling
    """
    # 1. Find molecules with this scaffold
    match_df = mol_df.query("Scaffold == @scaffold")
    
    # 2. Merge with activity data
    # Ensure we don't duplicate columns if they exist in both
    cols_to_merge = ["SMILES", "Name"]
    
    # We need pIC50 and mol from activity_df. 
    # Check what columns act_df has
    extra_cols = [c for c in ["pIC50", "mol"] if c in activity_df.columns]
    
    merge_df = match_df.merge(activity_df[cols_to_merge + extra_cols], on=["SMILES", "Name"], how="inner")
    
    # 3. Validation: Ensure pIC50 exists
    if "pIC50" not in merge_df.columns:
        # Check if merge created suffixes (e.g., pIC50_x, pIC50_y)
        suffixed_cols = [c for c in merge_df.columns if "pIC50" in c]
        if suffixed_cols:
            merge_df["pIC50"] = merge_df[suffixed_cols[0]]
        else:
            merge_df["pIC50"] = np.nan

    scaffold_mol = Chem.MolFromSmiles(scaffold)
    
    if scaffold_mol is None or merge_df.empty:
        return [], merge_df
        
    # 4. R-Group Decomposition
    try:
        # Ensure 'mol' column is valid RDKit objects
        mols = merge_df.mol.tolist()
        rgroup_match, rgroup_miss = RGroupDecompose(scaffold_mol, mols, asSmiles=True)
    except Exception as e:
        st.error(f"Decomposition failed: {e}")
        return [], merge_df

    if len(rgroup_match):
        rgroup_df = pd.DataFrame(rgroup_match)
        # Robustly get Core
        cores = rgroup_df.Core.unique() if 'Core' in rgroup_df.columns else []
        return cores, merge_df
    else:
        return [], merge_df

def mol_to_image_html(smiles):
    """Generate HTML image tag for a SMILES string"""
    if not smiles: return ""
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return "Invalid SMILES"
    img = Draw.MolToImage(mol, size=(250, 250))
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f'<img src="data:image/png;base64,{img_str}" style="border:1px solid #ddd; border-radius:5px;"/>'


# --- 2. Main Streamlit App ---

def main():
    st.set_page_config(page_title="MMP Scaffold Analysis", layout="wide", page_icon="🧪")
    
    # Styling
    st.markdown("""
        <style>
        .block-container {padding-top: 2rem;}
        h1 {color: #2e86de;}
        </style>
    """, unsafe_allow_html=True)

    st.title("🧪 MMP Scaffold Finder")
    st.markdown("""
    Identify common scaffolds and analyze activity distributions using Matched Molecular Pair (MMP) fragmentation rules.
    """)

    # --- Sidebar: Data Loading ---
    st.sidebar.header("📂 Data Upload")
    uploaded_file = st.sidebar.file_uploader("Upload SMILES or CSV", type=["csv", "smi", "txt"])
    
    if uploaded_file:
        try:
            # 1. Read File
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                # Robust read for SMI files
                # Try reading with header first
                df = pd.read_csv(uploaded_file, sep=r'\s+', engine='python')
                # If first column doesn't look like a header (e.g., contains 'c1ccccc1'), reload without header
                if len(df) > 0 and isinstance(df.iloc[0,0], str) and 'c' in df.iloc[0,0].lower():
                    # Likely a SMILES string in the first row, implying no header?
                    # Or check against common keywords
                    cols_str = str(list(df.columns)).lower()
                    if "smiles" not in cols_str and "pic50" not in cols_str:
                         uploaded_file.seek(0)
                         df = pd.read_csv(uploaded_file, sep=r'\s+', engine='python', header=None)
            
            # Clean column names
            if isinstance(df.columns[0], str):
                df.columns = df.columns.str.strip()
            df.columns = df.columns.astype(str)

        except Exception as e:
            st.error(f"Error reading file: {e}")
            return

        # 2. Column Mapping
        st.sidebar.subheader("Column Mapping")
        cols = df.columns.tolist()
        
        def get_idx(options, keywords):
            for i, opt in enumerate(options):
                if any(k in opt.lower() for k in keywords):
                    return i
            return 0

        # Smart defaults
        smi_default = get_idx(cols, ["smi", "structure", "can"])
        name_default = get_idx(cols, ["name", "id", "chembl", "compound"])
        if name_default == smi_default and len(cols) > 1: name_default = 1
        
        act_default = get_idx(cols, ["pic50", "activity", "val", "ic50"])
        if act_default == smi_default and len(cols) > 2: act_default = 2

        smi_col = st.sidebar.selectbox("SMILES Column", cols, index=smi_default)
        name_col = st.sidebar.selectbox("ID/Name Column", cols, index=name_default)
        act_col = st.sidebar.selectbox("pIC50 Column", cols, index=act_default)

        # 3. Prepare Data
        if st.sidebar.button("Run Analysis", type="primary"):
            with st.spinner("Initializing..."):
                try:
                    # Standardize DataFrame
                    input_df = df[[smi_col, name_col, act_col]].copy()
                    input_df.columns = ["SMILES", "Name", "pIC50"]
                    
                    # Force pIC50 to numeric
                    input_df["pIC50"] = pd.to_numeric(input_df["pIC50"], errors='coerce')
                    
                    # Calculate Mols
                    input_df['mol'] = input_df.SMILES.apply(Chem.MolFromSmiles)
                    
                    # Remove invalid
                    valid_mask = input_df['mol'].notnull()
                    if not valid_mask.all():
                        st.warning(f"Removed {len(input_df) - valid_mask.sum()} invalid SMILES")
                    input_df = input_df[valid_mask]
                    
                    if len(input_df) == 0:
                        st.error("No valid molecules found.")
                        st.stop()
                    
                    # Run Algorithm
                    mol_df, scaffold_df = find_scaffolds(input_df)
                    
                    # Store in Session State
                    st.session_state['data'] = {
                        'input_df': input_df,
                        'mol_df': mol_df,
                        'scaffold_df': scaffold_df
                    }
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    # Print full traceback for debugging
                    import traceback
                    st.code(traceback.format_exc())

    # --- Main Display ---
    if 'data' in st.session_state:
        data = st.session_state['data']
        scaffold_df = data['scaffold_df']
        mol_df = data['mol_df']
        input_df = data['input_df']

        # Layout
        tab1, tab2 = st.tabs(["📊 Scaffold Statistics", "🔍 Detailed Analysis"])

        with tab1:
            st.subheader(f"Found {len(scaffold_df)} Unique Scaffolds")
            
            # Filters
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                min_count = st.slider("Minimum Molecules per Scaffold", 1, int(scaffold_df['Count'].max()), 2)
            
            filtered_scaffolds = scaffold_df[scaffold_df['Count'] >= min_count]
            
            st.dataframe(
                filtered_scaffolds.style.background_gradient(subset=['Count'], cmap='Blues'),
                use_container_width=True,
                column_config={
                    "Scaffold": st.column_config.TextColumn("Scaffold SMILES", help="Copy this to view structure"),
                    "Count": st.column_config.NumberColumn("Frequency", format="%d")
                }
            )
            
            csv = filtered_scaffolds.to_csv(index=False).encode('utf-8')
            st.download_button("Download Statistics CSV", csv, "scaffold_stats.csv", "text/csv")

        with tab2:
            st.subheader("Deep Dive")
            
            # Select Scaffold
            scaffold_opts = filtered_scaffolds['Scaffold'].values
            if len(scaffold_opts) == 0:
                st.info("No scaffolds match filter criteria.")
            else:
                selected_scaffold = st.selectbox(
                    "Select Scaffold (Sorted by Frequency)", 
                    scaffold_opts,
                    format_func=lambda x: f"Freq: {filtered_scaffolds[filtered_scaffolds.Scaffold==x].Count.iloc[0]} | {x[:30]}..."
                )
                
                if selected_scaffold:
                    cores, matches_df = get_molecules_with_scaffold(selected_scaffold, mol_df, input_df)
                    
                    # Top Section: Structures
                    col_struct, col_stats = st.columns([1, 2])
                    
                    with col_struct:
                        st.markdown("**Scaffold Structure**")
                        st.markdown(mol_to_image_html(selected_scaffold), unsafe_allow_html=True)
                        
                        if len(cores) > 0:
                            st.markdown("**Core (with R-labels)**")
                            for c in cores:
                                st.markdown(mol_to_image_html(c), unsafe_allow_html=True)
                    
                    with col_stats:
                        st.markdown("**Activity Statistics**")
                        ic50_vals = matches_df['pIC50'].dropna()
                        
                        if len(ic50_vals) > 0:
                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("Count", len(matches_df))
                            m2.metric("Max pIC50", f"{ic50_vals.max():.2f}")
                            m3.metric("Min pIC50", f"{ic50_vals.min():.2f}")
                            m4.metric("Std Dev", f"{ic50_vals.std():.2f}")
                            
                            # Simple Chart
                            st.bar_chart(matches_df.set_index("Name")["pIC50"])
                        else:
                            st.warning("No valid numeric pIC50 values for this scaffold.")

                    # Bottom Section: Table
                    st.markdown("### Associated Molecules")
                    st.dataframe(
                        matches_df[["Name", "pIC50", "SMILES"]].sort_values("pIC50", ascending=False),
                        use_container_width=True
                    )

if __name__ == "__main__":
    main()
