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
# 1. Robust Helper Functions
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

def fragment_mol_robust(mol, use_relaxed_rules=False):
    """
    Perform single-cut fragmentation using RDKit's MMPA.
    Returns a list of (Core, R_group) pairs sorted by size.
    """
    if mol is None:
        return []
    
    # Strategy 1: Standard MMPA (Strict)
    # Target: Acyclic single bonds between specific Carbons
    try:
        if not use_relaxed_rules:
            frags = rdMMPA.FragmentMol(mol, minCuts=1, maxCuts=1, resultsAsMols=True)
        else:
            # Strategy 2: Relaxed MMPA
            # Target: ANY acyclic single bond ([*]!@-[*])
            # This is much more permissive and should work for almost any molecule with a sidechain
            relaxed_pat = Chem.MolFromSmarts('[*]!@-[*]')
            frags = rdMMPA.FragmentMol(mol, minCuts=1, maxCuts=1, pattern=relaxed_pat, resultsAsMols=True)
            
    except Exception:
        return []
    
    valid_pairs = []
    for core, sidechain in frags:
        if core is None or sidechain is None:
            continue
            
        m1 = Chem.Mol(core)
        m2 = Chem.Mol(sidechain)
        
        # Identify Core (larger) vs R-group (smaller)
        sorted_frags = sort_fragments([m1, m2])
        core_frag = sorted_frags[0]
        r_group_frag = sorted_frags[1]
        
        # Clean map numbers for cleaner SMILES
        remove_map_nums(core_frag)
        remove_map_nums(r_group_frag)
        
        valid_pairs.append((core_frag, r_group_frag))
        
    return valid_pairs

# ==========================================
# 2. Main Streamlit App
# ==========================================

st.set_page_config(page_title="MMP Analysis App", layout="wide")
st.title("🧪 Robust MMP Analysis")

# --- Sidebar ---
with st.sidebar:
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.dataframe(df.head(3))
        
        st.header("2. Settings")
        cols = df.columns.tolist()
        
        # Smart defaults
        default_smiles = next((c for c in cols if 'smi' in c.lower()), cols[0])
        default_name = next((c for c in cols if 'name' in c.lower() or 'id' in c.lower()), cols[0])
        default_prop = next((c for c in cols if 'pic50' in c.lower() or 'act' in c.lower() or 'val' in c.lower()), cols[0])

        smiles_col = st.selectbox("SMILES Column", cols, index=cols.index(default_smiles))
        name_col = st.selectbox("Name/ID Column", cols, index=cols.index(default_name))
        prop_col = st.selectbox("Property Column", cols, index=cols.index(default_prop))
        
        use_relaxed = st.checkbox("Use Relaxed Fragmentation Rules", value=True, 
                                help="If checked, cuts ANY acyclic single bond. If unchecked, uses strict Carbon-Carbon rules.")
        
        min_occurence = st.number_input("Min Transform Occurrence", min_value=2, value=5)
        run_btn = st.button("Run Analysis", type="primary")

# --- Analysis Logic ---
if uploaded_file and 'run_btn' in locals() and run_btn:
    st.info("Starting Analysis...")
    progress_bar = st.progress(0)
    
    # 1. Preprocessing
    input_df = df.copy().dropna(subset=[smiles_col, prop_col])
    
    st.write("Generating molecules...")
    input_df['mol'] = input_df[smiles_col].apply(lambda x: Chem.MolFromSmiles(str(x)))
    valid_mols_df = input_df[input_df['mol'].notnull()].copy()
    valid_mols_df['mol'] = valid_mols_df['mol'].apply(get_largest_fragment)
    
    if len(valid_mols_df) == 0:
        st.error("No valid molecules found. Check SMILES column.")
        st.stop()
        
    progress_bar.progress(20)
    
    # 2. Fragmentation Loop
    st.write(f"Fragmenting {len(valid_mols_df)} molecules (Relaxed Rules: {use_relaxed})...")
    
    row_list = []
    debug_failures = []
    
    for idx, row in valid_mols_df.iterrows():
        mol = row['mol']
        # Try fragmentation
        pairs = fragment_mol_robust(mol, use_relaxed_rules=use_relaxed)
        
        if not pairs:
            # If standard failed and we aren't already relaxed, try relaxed as fallback? 
            # (Optional, but let's stick to user choice for now)
            debug_failures.append(row[smiles_col])
            continue
            
        for core, r_group in pairs:
            core_smi = Chem.MolToSmiles(core)
            r_smi = Chem.MolToSmiles(r_group)
            row_list.append([row[smiles_col], core_smi, r_smi, str(row[name_col]), row[prop_col]])
            
    progress_bar.progress(50)
    
    # 3. Diagnostic Expander (Vital for debugging)
    with st.expander("Fragmentation Diagnostics", expanded=False):
        st.write(f"Successful Fragments: {len(row_list)}")
        st.write(f"Failed Molecules: {len(debug_failures)}")
        if len(debug_failures) > 0:
            st.warning("First 5 molecules that failed to fragment:")
            for fail_smi in debug_failures[:5]:
                st.code(fail_smi)
                st.caption("Reason: No matching bonds found for current rules.")
    
    if len(row_list) == 0:
        st.error("❌ Still no fragments generated! Try checking 'Use Relaxed Fragmentation Rules' in the sidebar.")
        st.stop()
    
    # 4. Pair Generation
    row_df = pd.DataFrame(row_list, columns=["SMILES", "Core", "R_group", "Name", "Property"])
    delta_list = []
    
    # Group by Core
    grouped = row_df.groupby("Core")
    for core_smi, group in grouped:
        if len(group) > 1:
            group_recs = group.to_dict('records')
            for r_a, r_b in combinations(group_recs, 2):
                if r_a['SMILES'] == r_b['SMILES']: continue
                
                # Sort A->B
                pair = sorted([r_a, r_b], key=lambda x: x['SMILES'])
                reagent_a, reagent_b = pair[0], pair[1]
                
                try:
                    delta = float(reagent_b['Property']) - float(reagent_a['Property'])
                except:
                    continue
                
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
    
    # 5. Results
    mmp_list = []
    if not delta_df.empty:
        for trans, group in delta_df.groupby("Transform"):
            if len(group) >= min_occurence:
                mmp_list.append([trans, len(group), group['Delta'].values, group['Delta'].mean()])
    
    mmp_df = pd.DataFrame(mmp_list, columns=["Transform", "Count", "Deltas", "Mean_Delta"])
    if not mmp_df.empty:
        mmp_df = mmp_df.sort_values("Mean_Delta", ascending=False).reset_index(drop=True)
    
    progress_bar.progress(100)
    
    if mmp_df.empty:
        st.warning(f"Fragments generated, but no matched pairs met the occurrence threshold ({min_occurence}).")
        st.write(f"Total pairs found: {len(delta_df)}")
    else:
        st.success(f"Found {len(mmp_df)} distinct transforms.")
        
        tab1, tab2, tab3 = st.tabs(["Overview", "Table", "Pairs"])
        
        with tab1:
            st.subheader("Distribution")
            top_n = st.slider("Show Top N", 5, 50, 15)
            subset_df = mmp_df.head(top_n)
            
            # Prepare plot data
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
                st.write(f"Showing {len(subset)} pairs")
                
                # Limit to first 20 for performance
                for i, row in subset.head(20).iterrows():
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(Draw.MolToImage(Chem.MolFromSmiles(row['SMILES_1']), size=(200,150)))
                        st.caption(f"{row['Name_1']} ({row['Property_1']})")
                    with c2:
                        st.image(Draw.MolToImage(Chem.MolFromSmiles(row['SMILES_2']), size=(200,150)))
                        st.caption(f"{row['Name_2']} ({row['Property_2']})")
                    st.divider()
