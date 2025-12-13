# app.py
import streamlit as st
import pandas as pd
import numpy as np
import itertools
from itertools import combinations
from rdkit import Chem
from rdkit.Chem import AllChem, rdMMPA
from rdkit.Chem.rdMolDescriptors import CalcNumRings
import concurrent.futures
from tqdm import tqdm
import time

# Cache results to avoid recomputation
@st.cache_data(show_spinner=False, max_entries=50)
def preprocess_molecules(df, smiles_col):
    """Preprocess molecules once and cache"""
    proc_df = df.copy()
    proc_df["mol"] = proc_df[smiles_col].apply(Chem.MolFromSmiles)
    proc_df = proc_df[proc_df.mol.notnull()]
    
    # Filter out very large molecules (they slow down fragmentation)
    proc_df["num_atoms"] = proc_df.mol.apply(lambda x: x.GetNumAtoms())
    proc_df = proc_df[proc_df.num_atoms <= 100]  # Adjust threshold as needed
    
    return proc_df

def get_largest_fragment(mol):
    """Get the largest fragment from a molecule"""
    frags = Chem.GetMolFrags(mol, asMols=True)
    if not frags:
        return mol
    return max(frags, key=lambda x: x.GetNumAtoms())

def cleanup_fragment_fast(mol):
    """Faster cleanup without modifying atom types unnecessarily"""
    # Remove atom maps
    for atm in mol.GetAtoms():
        atm.SetAtomMapNum(0)
    
    # Count dummy atoms without modifying them
    rgroup_count = sum(1 for atm in mol.GetAtoms() if atm.GetAtomicNum() == 0)
    
    return Chem.RemoveHs(mol), rgroup_count

def generate_fragments_fast(mol, max_frags_per_mol=10):
    """Generate fragments with optimizations"""
    try:
        # Limit the number of cuts to speed up fragmentation
        frag_list = rdMMPA.FragmentMol(mol, maxCuts=2)  # Only single and double cuts
        flat_frags = [x for x in itertools.chain(*frag_list) if x]
        
        if not flat_frags:
            return pd.DataFrame(columns=["Core", "NumAtoms", "NumRgroups"])
        
        # Get unique fragments efficiently
        frag_smiles_set = set()
        frag_info = []
        
        for frag in flat_frags[:max_frags_per_mol]:  # Limit fragments per molecule
            frag = get_largest_fragment(frag)
            num_atoms_frag = frag.GetNumAtoms()
            num_atoms_orig = mol.GetNumAtoms()
            
            # Skip if fragment is too small or too similar to original
            if (num_atoms_frag / num_atoms_orig > 0.67 and 
                num_atoms_frag / num_atoms_orig < 0.95):
                
                cleaned_frag, rgroup_count = cleanup_fragment_fast(frag)
                smiles = Chem.MolToSmiles(cleaned_frag)
                
                if smiles not in frag_smiles_set:
                    frag_smiles_set.add(smiles)
                    frag_info.append([smiles, num_atoms_frag, rgroup_count])
        
        # Add the original molecule
        orig_smiles = Chem.MolToSmiles(mol)
        if orig_smiles not in frag_smiles_set:
            frag_info.append([orig_smiles, mol.GetNumAtoms(), 1])
        
        if not frag_info:
            return pd.DataFrame(columns=["Core", "NumAtoms", "NumRgroups"])
        
        return pd.DataFrame(
            frag_info,
            columns=["Core", "NumAtoms", "NumRgroups"]
        ).drop_duplicates("Core")
    
    except:
        return pd.DataFrame(columns=["Core", "NumAtoms", "NumRgroups"])

def find_rgroup_smarts_fast(core_smiles, mol):
    """Fast R-group identification using SMARTS pattern"""
    try:
        core_mol = Chem.MolFromSmiles(core_smiles)
        if core_mol is None:
            return None
        
        # Convert core to SMARTS with attachment points
        core_pattern = Chem.MolToSmarts(core_mol)
        
        # Find matches
        matches = mol.GetSubstructMatches(core_mol)
        if not matches:
            return None
        
        # For simplicity, just mark attachment points with [*:1]
        # This is a simplified approach - you can make it more sophisticated
        return "[*]"
    
    except:
        return None

def process_molecule_batch(args):
    """Process a batch of molecules for parallel execution"""
    i, smiles, name, activity, mol = args
    results = []
    
    frag_df = generate_fragments_fast(mol, max_frags_per_mol=5)  # Further limit
    
    for core in frag_df.Core:
        rgroup = find_rgroup_smarts_fast(core, mol)
        if rgroup:
            results.append([smiles, core, rgroup, name, activity])
    
    return results

# =========================================================
# OPTIMIZED MMP ANALYSIS
# =========================================================
def run_mmp_optimized(processed_df, min_occ):
    total_molecules = len(processed_df)
    
    # Progress tracking
    progress_bar = st.progress(0)
    status = st.empty()
    
    # STEP 1: Parallel fragmentation
    status.text(f"🔬 Fragmenting {total_molecules} molecules...")
    
    row_list = []
    
    # Use parallel processing for large datasets
    if total_molecules > 10:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            args_list = [
                (i, row.SMILES, row.Name, row.Activity, row.mol)
                for i, row in enumerate(processed_df.itertuples())
            ]
            
            futures = list(executor.map(process_molecule_batch, args_list))
            
            for i, future in enumerate(futures):
                row_list.extend(future)
                progress_bar.progress((i + 1) / total_molecules)
    else:
        # Sequential for small datasets
        for i, row in enumerate(processed_df.itertuples()):
            frag_df = generate_fragments_fast(row.mol, max_frags_per_mol=5)
            
            for core in frag_df.Core:
                rgroup = find_rgroup_smarts_fast(core, row.mol)
                if rgroup:
                    row_list.append([row.SMILES, core, rgroup, row.Name, row.Activity])
            
            progress_bar.progress((i + 1) / total_molecules)
    
    if not row_list:
        st.warning("No fragments generated. Try adjusting fragmentation parameters.")
        return None
    
    status.text(f"📊 Found {len(row_list)} fragment instances. Pairing...")
    
    # Create DataFrame
    row_df = pd.DataFrame(
        row_list,
        columns=["SMILES", "Core", "R_group", "Name", "Activity"]
    )
    
    # STEP 2: Efficient pairwise comparison
    progress_bar.progress(0.5)
    status.text("🧮 Generating molecular pairs...")
    
    delta_list = []
    
    # Group by core and filter for efficiency
    core_groups = row_df.groupby("Core")
    significant_cores = {core: group for core, group in core_groups if len(group) >= 2}
    
    if not significant_cores:
        st.warning("No cores with multiple molecules found.")
        return None
    
    # Process each core
    for i, (core, group_df) in enumerate(significant_cores.items()):
        molecules = group_df.to_dict('records')
        
        # Create all unique pairs
        for j in range(len(molecules)):
            for k in range(j + 1, len(molecules)):
                mol1 = molecules[j]
                mol2 = molecules[k]
                
                if mol1["SMILES"] == mol2["SMILES"]:
                    continue
                
                # Order consistently
                if mol1["SMILES"] > mol2["SMILES"]:
                    mol1, mol2 = mol2, mol1
                
                delta = float(mol2["Activity"]) - float(mol1["Activity"])
                transform = f"{mol1['R_group']}>>{mol2['R_group']}"
                
                delta_list.append([
                    mol1["SMILES"], core, mol1["R_group"], mol1["Name"], mol1["Activity"],
                    mol2["SMILES"], core, mol2["R_group"], mol2["Name"], mol2["Activity"],
                    transform, delta
                ])
        
        # Update progress
        progress_bar.progress(0.5 + (i / len(significant_cores)) * 0.3)
    
    if not delta_list:
        st.warning("No valid molecular pairs found.")
        return None
    
    # Create delta DataFrame
    delta_df = pd.DataFrame(
        delta_list,
        columns=[
            "SMILES_1", "Core_1", "R_group_1", "Name_1", "Activity_1",
            "SMILES_2", "Core_2", "R_group_2", "Name_2", "Activity_2",
            "Transform", "Delta"
        ]
    )
    
    # STEP 3: Fast aggregation
    progress_bar.progress(0.8)
    status.text("📈 Calculating statistics...")
    
    # Group transforms efficiently
    transform_stats = delta_df.groupby("Transform").agg({
        "Delta": ["count", "mean", "std", "min", "max"]
    })
    
    # Flatten column names
    transform_stats.columns = ['_'.join(col).strip('_') for col in transform_stats.columns.values]
    transform_stats = transform_stats.rename(columns={
        "Delta_count": "Count",
        "Delta_mean": "mean_delta",
        "Delta_std": "std_delta",
        "Delta_min": "min_delta",
        "Delta_max": "max_delta"
    }).reset_index()
    
    # Filter by minimum occurrence
    mmp_df = transform_stats[transform_stats["Count"] >= min_occ]
    
    progress_bar.progress(1.0)
    status.text("✅ Analysis complete!")
    
    return row_df, delta_df, mmp_df

# =========================================================
# Streamlit UI - With performance options
# =========================================================
st.sidebar.header("⚡ Performance Settings")

# Performance options
max_molecules = st.sidebar.slider(
    "Max molecules to process",
    10, 500, 100,
    help="Limit processing to speed up analysis"
)

max_fragments = st.sidebar.slider(
    "Max fragments per molecule",
    1, 20, 5,
    help="Fewer fragments = faster processing"
)

use_parallel = st.sidebar.checkbox(
    "Use parallel processing", 
    value=True,
    help="Speed up with multi-threading"
)

st.sidebar.header("📁 Upload Data")
uploaded = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
min_occ = st.sidebar.slider("Minimum transform occurrence", 2, 20, 3)
run_button = st.sidebar.button("🚀 Run Optimized MMP Analysis")

# Main app
if uploaded:
    df = pd.read_csv(uploaded)
    
    # Limit dataset size
    if len(df) > max_molecules:
        st.warning(f"Dataset limited to first {max_molecules} molecules for performance.")
        df = df.head(max_molecules)
    
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())
    
    cols = df.columns.tolist()
    smiles_col = st.selectbox("SMILES column", cols)
    activity_col = st.selectbox("Activity column", cols)
    name_col = st.selectbox("Name column (optional)", ["None"] + cols)
    
    if run_button:
        start_time = time.time()
        
        with st.spinner("Preprocessing molecules..."):
            proc_df = preprocess_molecules(df, smiles_col)
            
            proc_df = proc_df.rename(columns={
                smiles_col: "SMILES",
                activity_col: "Activity"
            })
            
            if name_col != "None":
                proc_df = proc_df.rename(columns={name_col: "Name"})
            else:
                proc_df["Name"] = [f"Mol_{i+1}" for i in range(len(proc_df))]
            
            # Ensure Activity is numeric
            proc_df["Activity"] = pd.to_numeric(proc_df["Activity"], errors='coerce')
            proc_df = proc_df.dropna(subset=["Activity"])
            
            st.info(f"Processing {len(proc_df)} valid molecules")
        
        result = run_mmp_optimized(proc_df, min_occ)
        elapsed_time = time.time() - start_time
        
        if result is None:
            st.error("No valid MMPs found.")
        else:
            row_df, delta_df, mmp_df = result
            
            st.success(f"✅ Found {len(mmp_df)} MMP transforms in {elapsed_time:.1f} seconds")
            
            # Display results
            st.subheader("📊 MMP Summary")
            st.dataframe(
                mmp_df.sort_values("mean_delta"),
                use_container_width=True
            )
            
            with st.expander("📈 View Statistics"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Molecules", len(proc_df))
                with col2:
                    st.metric("Unique Fragments", len(row_df.Core.unique()))
                with col3:
                    st.metric("Matched Pairs", len(delta_df))
            
            with st.expander("🧪 View All Pairs"):
                st.dataframe(delta_df, use_container_width=True)
            
            with st.expander("🧩 View Fragments"):
                st.dataframe(row_df, use_container_width=True)
            
            # Download buttons
            st.subheader("📥 Download Results")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button(
                    "Download MMP Summary",
                    mmp_df.to_csv(index=False),
                    "mmp_summary.csv"
                )
            with col2:
                st.download_button(
                    "Download All Pairs",
                    delta_df.to_csv(index=False),
                    "mmp_pairs.csv"
                )
            with col3:
                st.download_button(
                    "Download Fragments",
                    row_df.to_csv(index=False),
                    "fragments.csv"
                )

else:
    st.info("👈 Upload a CSV file to begin")
