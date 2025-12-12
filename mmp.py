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

# ==========================================
# 1. Helper Functions
# ==========================================

def get_largest_fragment(mol):
    """Returns the largest fragment of a molecule."""
    if mol is None:
        return None
    try:
        lfc = rdMolStandardize.LargestFragmentChooser()
        return lfc.choose(mol)
    except:
        return mol

def remove_map_nums(mol):
    """Remove atom map numbers from a molecule (in-place)."""
    for atm in mol.GetAtoms():
        atm.SetAtomMapNum(0)

def sort_fragments(mol_list):
    """Sort a list of molecules by number of atoms (largest to smallest)."""
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
    
    # Generate cuts (returns list of tuples: (core, sidechain))
    try:
        frags = rdMMPA.FragmentMol(mol, maxCuts=1, resultsAsMols=True)
    except Exception:
        return []
    
    valid_pairs = []
    for core, sidechain in frags:
        if core is None or sidechain is None:
            continue
            
        m1 = Chem.Mol(core)
        m2 = Chem.Mol(sidechain)
        
        # Sort to identify Core (larger) vs R-group (smaller)
        sorted_frags = sort_fragments([m1, m2])
        core_frag = sorted_frags[0]
        r_group_frag = sorted_frags[1]
        
        # Clean map numbers
        remove_map_nums(core_frag)
        remove_map_nums(r_group_frag)
        
        valid_pairs.append((core_frag, r_group_frag))
        
    return valid_pairs

# ==========================================
# 2. Main Streamlit App
# ==========================================

st.set_page_config(page_title="MMP Analysis App", layout="wide")
st.title("🧪 Matched Molecular Pair (MMP) Analysis")

# --- Sidebar ---
with st.sidebar:
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file:
        # Load data
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(df)} rows.")
            st.write("Data Preview:")
            st.dataframe(df.head(3))
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            st.stop()
        
        st.header("2. Map Columns")
        cols = df.columns.tolist()
        
        # Smart defaults
        default_smiles = next((c for c in cols if 'smi' in c.lower()), cols[0])
        default_name = next((c for c in cols if 'name' in c.lower() or 'id' in c.lower()), cols[0])
        default_prop = next((c for c in cols if 'pic50' in c.lower() or 'act' in c.lower() or 'val' in c.lower()), cols[0])

        smiles_col = st.selectbox("SMILES Column", cols, index=cols.index(default_smiles))
        name_col = st.selectbox("Name/ID Column", cols, index=cols.index(default_name))
        prop_col = st.selectbox("Property Column (e.g. pIC50)", cols, index=cols.index(default_prop))
        
        min_occurence = st.number_input("Min Transform Occurrence", min_value=2, value=5)
        run_btn = st.button("Run Analysis", type="primary")

# --- Analysis Logic ---
if uploaded_file and 'run_btn' in locals() and run_btn:
    st.info("Starting Analysis...")
    progress_bar = st.progress(0)
    
    # 1. Preprocessing with Diagnostics
    input_df = df.copy()
    initial_count = len(input_df)
    
    # Step A: Check for empty values
    input_df = input_df.dropna(subset=[smiles_col, prop_col])
    cleaned_count = len(input_df)
    
    if cleaned_count == 0:
        st.error(f"❌ Error: All rows were dropped. Check if columns '{smiles_col}' or '{prop_col}' are empty.")
        st.stop()
    elif cleaned_count < initial_count:
        st.warning(f"⚠️ Dropped {initial_count - cleaned_count} rows due to missing SMILES or Property values.")

    # Step B: Generate Molecules
    st.write("Generating RDKit molecules...")
    input_df['mol'] = input_df[smiles_col].apply(lambda x: Chem.MolFromSmiles(str(x)))
    
    # Step C: Filter Invalid Molecules
    valid_mols_df = input_df[input_df['mol'].notnull()].copy()
    valid_count = len(valid_mols_df)
    
    if valid_count == 0:
        st.error(f"❌ Error: Could not parse any SMILES strings from column '{smiles_col}'. Please verify your column selection.")
        st.stop()
    elif valid_count < cleaned_count:
        st.warning(f"⚠️ {cleaned_count - valid_count} SMILES strings could not be parsed and were skipped.")
    
    valid_mols_df['mol'] = valid_mols_df['mol'].apply(get_largest_fragment)
    progress_bar.progress(20)
    
    # 2. Fragmentation Loop
    st.write(f"Fragmenting {valid_count} molecules...")
    row_list = []
    
    for idx, row in valid_mols_df.iterrows():
        mol = row['mol']
        name = str(row[name_col])
        val = row[prop_col]
        smi = row[smiles_col]
        
        # Fragment
        pairs = fragment_mol_simple(mol)
        
        for core, r_group in pairs:
            core_smi = Chem.MolToSmiles(core)
            r_smi = Chem.MolToSmiles(r_group)
            row_list.append([smi, core_smi, r_smi, name, val])
            
    progress_bar.progress(50)
    
    # 3. Validation Check
    if len(row_list) == 0:
        st.error("❌ No fragments were generated!")
        st.markdown("""
        **Possible reasons:**
        1. **Molecules are too small:** Single cut fragmentation requires acyclic single bonds. Small molecules like Methane or Benzene (rigid rings) cannot be cut.
        2. **No rotatable bonds:** The algorithm looks for single bonds not in rings.
        """)
        st.stop()
    
    st.success(f"Generated {len(row_list)} fragments from {valid_count} molecules.")
    row_df = pd.DataFrame(row_list, columns=["SMILES", "Core", "R_group", "Name", "Property"])
    
    # 4. Pair Generation
    delta_list = []
    grouped = row_df.groupby("Core")
    
    for core_smi, group in grouped:
        if len(group) > 1:
            group_recs = group.to_dict('records')
            for r_a, r_b in combinations(group_recs, 2):
                if r_a['SMILES'] == r_b['SMILES']: continue
                
                pair = sorted([r_a, r_b], key=lambda x: x['SMILES'])
                reagent_a, reagent_b = pair[0], pair[1]
                
                # Check property type
                try:
                    delta = float(reagent_b['Property']) - float(reagent_a['Property'])
                except:
                    continue # Skip if property is not numeric
                
                trans_str = f"{reagent_a['R_group'].replace('*','*-')}>>{reagent_b['R_group'].replace('*','*-')}"
                
                delta_list.append([
                    reagent_a['SMILES'], reagent_a['Core'], reagent_a['R_group'], reagent_a['Name'], reagent_a['Property'],
                    reagent_b['SMILES'], reagent_b['Core'], reagent_b['R_group'], reagent_b['Name'], reagent_b['Property'],
                    trans_str, delta
                ])
                
    delta_df = pd.DataFrame(delta_list, columns=[
        "SMILES_1","Core_1","R_group_1","Name_1","Property_1",
        "SMILES_2","Core_2","R_group_2","Name_2","Property_2",
        "Transform","Delta"])
    
    progress_bar.progress(80)
    
    # 5. MMP Stats
    mmp_list = []
    if not delta_df.empty:
        for trans, group in delta_df.groupby("Transform"):
            if len(group) >= min_occurence:
                mmp_list.append([trans, len(group), group['Delta'].values, group['Delta'].mean()])
    
    mmp_df = pd.DataFrame(mmp_list, columns=["Transform", "Count", "Deltas", "Mean_Delta"])
    if not mmp_df.empty:
        mmp_df = mmp_df.sort_values("Mean_Delta", ascending=False).reset_index(drop=True)
    
    progress_bar.progress(100)
    
    # --- Results Display ---
    if mmp_df.empty:
        st.warning("No matched molecular pairs found that meet the occurrence threshold.")
        st.write(f"Total pairs identified: {len(delta_df)}")
        if len(delta_df) > 0:
            st.write("Try lowering the 'Min Transform Occurrence' setting.")
    else:
        st.success(f"Found {len(mmp_df)} distinct transforms.")
        
        tab1, tab2, tab3 = st.tabs(["Overview", "Table", "Pairs"])
        
        with tab1:
            st.subheader("Distribution")
            top_n = 20
            subset_df = mmp_df.head(top_n)
            plot_data = []
            for idx, row in subset_df.iterrows():
                for d in row['Deltas']:
                    plot_data.append({'Transform': row['Transform'], 'Delta': d})
            
            if plot_data:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(data=pd.DataFrame(plot_data), x='Delta', y='Transform', ax=ax, orient='h')
                plt.axvline(0, color='r', linestyle='--')
                st.pyplot(fig)
        
        with tab2:
            st.dataframe(mmp_df[['Transform', 'Count', 'Mean_Delta']].style.background_gradient(subset=['Mean_Delta'], cmap='coolwarm'))

        with tab3:
            selected_trans = st.selectbox("Select Transform", mmp_df['Transform'].tolist())
            if selected_trans:
                subset = delta_df[delta_df['Transform'] == selected_trans]
                for i, row in subset.iterrows():
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(Draw.MolToImage(Chem.MolFromSmiles(row['SMILES_1']), size=(200,150)))
                        st.caption(f"{row['Name_1']} ({row['Property_1']})")
                    with c2:
                        st.image(Draw.MolToImage(Chem.MolFromSmiles(row['SMILES_2']), size=(200,150)))
                        st.caption(f"{row['Name_2']} ({row['Property_2']})")
                    st.divider()
