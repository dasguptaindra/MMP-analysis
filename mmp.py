import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem import rdMMPA
from rdkit.Chem.MolStandardize import rdMolStandardize
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# ==========================================
# 1. Helper Functions (Refactored for Standalone Use)
# ==========================================

def get_largest_fragment(mol):
    """
    Returns the largest fragment of a molecule.
    Replaces useful_rdkit_utils.get_largest_fragment
    """
    if mol is None:
        return None
    try:
        lfc = rdMolStandardize.LargestFragmentChooser()
        return lfc.choose(mol)
    except:
        return mol

def remove_map_nums(mol):
    """
    Remove atom map numbers from a molecule (in-place).
    """
    for atm in mol.GetAtoms():
        atm.SetAtomMapNum(0)

def sort_fragments(mol_list):
    """
    Sort a list of molecules by number of atoms (largest to smallest).
    """
    frag_num_atoms_list = [(x.GetNumAtoms(), x) for x in mol_list]
    frag_num_atoms_list.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in frag_num_atoms_list]

def fragment_mol_simple(mol):
    """
    Perform single-cut fragmentation using RDKit's MMPA.
    Returns a list of (Core, R_group) pairs sorted by size.
    """
    if mol is None:
        return []
    
    # maxCuts=1 generates single cuts (producing 2 fragments)
    # rdMMPA.FragmentMol returns a list of tuples: (core, sidechain)
    frags = rdMMPA.FragmentMol(mol, maxCuts=1, resultsAsMols=True)
    
    valid_pairs = []
    for core, sidechain in frags:
        if core is None or sidechain is None:
            continue
            
        # The user logic treats the larger piece as 'Core' and smaller as 'R_group'
        # We clean them (remove map nums) first? 
        # Note: MMPA adds map nums (dummy labels). We usually keep them to see attachment points
        # but the user's code removed them for the final SMILES in some places, 
        # yet the Transform string "A>>B" implies attachment points might be needed.
        # The user's code: remove_map_nums(x) was called on the pair list.
        # We will clone them to avoid modifying the originals if needed.
        
        m1 = Chem.Mol(core)
        m2 = Chem.Mol(sidechain)
        
        # Sort to identify Core vs R-group
        sorted_frags = sort_fragments([m1, m2])
        core_frag = sorted_frags[0]
        r_group_frag = sorted_frags[1]
        
        # Remove map numbers for canonical SMILES generation if desired, 
        # but usually we want to keep the dummy atom (wildcard) to indicate attachment.
        # The user code had: remove_map_nums(x) for x in frag_list
        # and then generated SMILES.
        remove_map_nums(core_frag)
        remove_map_nums(r_group_frag)
        
        valid_pairs.append((core_frag, r_group_frag))
        
    return valid_pairs

# ==========================================
# 2. Main Streamlit App
# ==========================================

st.set_page_config(page_title="MMP Analysis App", layout="wide")
st.title("🧪 Matched Molecular Pair (MMP) Analysis")
st.markdown("""
This app performs MMP analysis to identify structural transformations and their effect on a property (e.g., pIC50).
Upload a CSV file containing molecules and activity data.
""")

# --- Sidebar: Upload & Settings ---
with st.sidebar:
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.dataframe(df.head(3))
        
        st.header("2. Map Columns")
        cols = df.columns.tolist()
        smiles_col = st.selectbox("SMILES Column", cols, index=cols.index('SMILES') if 'SMILES' in cols else 0)
        name_col = st.selectbox("Name/ID Column", cols, index=cols.index('Name') if 'Name' in cols else 0)
        prop_col = st.selectbox("Property Column (e.g. pIC50)", cols, index=cols.index('pIC50') if 'pIC50' in cols else 0)
        
        min_occurence = st.number_input("Min Transform Occurrence", min_value=2, value=5)
        
        run_btn = st.button("Run Analysis", type="primary")

# --- Main Analysis Logic ---
if uploaded_file and 'run_btn' in locals() and run_btn:
    st.info("Running MMP Analysis... This may take a moment.")
    
    progress_bar = st.progress(0)
    
    # 1. Preprocessing
    input_df = df.copy()
    input_df = input_df.dropna(subset=[smiles_col, prop_col])
    
    # Add Mol column
    input_df['mol'] = input_df[smiles_col].apply(Chem.MolFromSmiles)
    # Filter invalid mols
    input_df = input_df[input_df['mol'].notnull()]
    # Get largest fragment
    input_df['mol'] = input_df['mol'].apply(get_largest_fragment)
    
    progress_bar.progress(20)
    
    # 2. Fragmentation Loop
    row_list = []
    
    # Iterate over rows
    total_rows = len(input_df)
    for idx, row in input_df.iterrows():
        mol = row['mol']
        name = row[name_col]
        val = row[prop_col]
        smi = row[smiles_col]
        
        # Fragment
        pairs = fragment_mol_simple(mol)
        
        for core, r_group in pairs:
            # Generate SMILES
            core_smi = Chem.MolToSmiles(core)
            r_smi = Chem.MolToSmiles(r_group)
            
            row_list.append([smi, core_smi, r_smi, name, val])
    
    row_df = pd.DataFrame(row_list, columns=["SMILES", "Core", "R_group", "Name", "Property"])
    
    progress_bar.progress(50)
    
    if len(row_df) == 0:
        st.error("No fragments generated. Check input structures.")
        st.stop()
        
    # 3. Pair Generation (Delta calculation)
    delta_list = []
    
    # Group by Core to find molecules sharing the same scaffold
    grouped = row_df.groupby("Core")
    
    for core_smi, group in grouped:
        if len(group) > 1:
            # Generate combinations of 2
            # We convert to list of dicts for easier iteration
            group_recs = group.to_dict('records')
            
            for r_a, r_b in combinations(group_recs, 2):
                if r_a['SMILES'] == r_b['SMILES']:
                    continue
                
                # Sort pair by SMILES to maintain consistent direction A->B vs B->A
                # (Matches user logic: sorted by SMILES key)
                pair = sorted([r_a, r_b], key=lambda x: x['SMILES'])
                reagent_a, reagent_b = pair[0], pair[1]
                
                delta = reagent_b['Property'] - reagent_a['Property']
                
                # Create transform string (with wildcards replacement for readability)
                # The user code replaced '*' with '*-' to make it look like a connection point
                trans_str = f"{reagent_a['R_group'].replace('*','*-')}>>{reagent_b['R_group'].replace('*','*-')}"
                
                # Store result
                # Columns: SMILES_1, Core, R_1, Name_1, Prop_1, SMILES_2, Core, R_2, Name_2, Prop_2, Transform, Delta
                delta_list.append([
                    reagent_a['SMILES'], reagent_a['Core'], reagent_a['R_group'], reagent_a['Name'], reagent_a['Property'],
                    reagent_b['SMILES'], reagent_b['Core'], reagent_b['R_group'], reagent_b['Name'], reagent_b['Property'],
                    trans_str, delta
                ])
                
    delta_cols = ["SMILES_1","Core_1","R_group_1","Name_1","Property_1",
                  "SMILES_2","Core_2","R_group_2","Name_2","Property_2",
                  "Transform","Delta"]
    delta_df = pd.DataFrame(delta_list, columns=delta_cols)
    
    progress_bar.progress(80)
    
    # 4. Aggregation (MMP Table)
    mmp_list = []
    for trans, group in delta_df.groupby("Transform"):
        if len(group) >= min_occurence:
            mmp_list.append([trans, len(group), group['Delta'].values, group['Delta'].mean()])
            
    mmp_df = pd.DataFrame(mmp_list, columns=["Transform", "Count", "Deltas", "Mean_Delta"])
    mmp_df = mmp_df.sort_values("Mean_Delta", ascending=False)
    mmp_df = mmp_df.reset_index(drop=True)
    
    progress_bar.progress(100)
    
    # --- Results Display ---
    
    st.success(f"Analysis Complete! Found {len(mmp_df)} distinct transforms occurring >= {min_occurence} times.")
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["Overview & Plots", "MMP Table", "Detailed Pairs"])
    
    with tab1:
        st.subheader("Distribution of Effects")
        if not mmp_df.empty:
            # Plot the top transforms
            top_n = st.slider("Show Top N Transforms", 5, 50, 20)
            
            # Prepare data for plotting (explode the Deltas list)
            plot_data = []
            # Sort by absolute mean delta for interesting plots? Or just Mean Delta?
            # Let's use the sorted mmp_df (by mean delta)
            subset_df = mmp_df.head(top_n) if len(mmp_df) > top_n else mmp_df
            
            for idx, row in subset_df.iterrows():
                for d in row['Deltas']:
                    plot_data.append({'Transform': row['Transform'], 'Delta': d})
            
            plot_df = pd.DataFrame(plot_data)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=plot_df, x='Delta', y='Transform', ax=ax, orient='h')
            plt.axvline(0, color='red', linestyle='--', alpha=0.5)
            plt.title("Property Delta Distribution by Transform")
            st.pyplot(fig)
        else:
            st.warning("No transforms met the occurrence threshold.")
            
    with tab2:
        st.subheader("MMP Summary Table")
        st.dataframe(
            mmp_df[['Transform', 'Count', 'Mean_Delta']].style.background_gradient(subset=['Mean_Delta'], cmap='coolwarm'),
            use_container_width=True
        )
        
    with tab3:
        st.subheader("Drill Down")
        selected_trans = st.selectbox("Select a Transform to View Pairs", mmp_df['Transform'].tolist())
        
        if selected_trans:
            subset = delta_df[delta_df['Transform'] == selected_trans]
            st.write(f"Showing {len(subset)} pairs for: **{selected_trans}**")
            
            # Show pairs as images + data
            for i, row in subset.iterrows():
                c1, c2 = st.columns(2)
                
                with c1:
                    st.caption(f"{row['Name_1']} (Val: {row['Property_1']})")
                    mol1 = Chem.MolFromSmiles(row['SMILES_1'])
                    if mol1:
                        img1 = Draw.MolToImage(mol1, size=(250, 200))
                        st.image(img1)
                    st.markdown(f"**R1:** `{row['R_group_1']}`")
                        
                with c2:
                    st.caption(f"{row['Name_2']} (Val: {row['Property_2']})")
                    mol2 = Chem.MolFromSmiles(row['SMILES_2'])
                    if mol2:
                        img2 = Draw.MolToImage(mol2, size=(250, 200))
                        st.image(img2)
                    st.markdown(f"**R2:** `{row['R_group_2']}`")
                
                st.metric("Delta", f"{row['Delta']:.3f}")
                st.divider()

else:
    if not uploaded_file:
        st.info("Please upload a CSV file to start.")

