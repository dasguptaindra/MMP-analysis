# app.py
import streamlit as st
import pandas as pd
import numpy as np
import itertools
from itertools import combinations
from rdkit import Chem
from rdkit.Chem import AllChem, rdMMPA, rdRGroupDecomposition
from rdkit.Chem.Fragments import fr_Al_OH, fr_Al_OH_noTert
from rdkit.Chem.Scaffolds import MurckoScaffold

# =========================================================
# Helper functions - CORRECTED VERSION
# =========================================================
def cleanup_fragment(mol):
    rgroup_count = 0
    for atm in mol.GetAtoms():
        atm.SetAtomMapNum(0)
        if atm.GetAtomicNum() == 0:
            rgroup_count += 1
            atm.SetAtomicNum(1)  # Replace dummy with hydrogen
    mol = Chem.RemoveAllHs(mol)
    return mol, rgroup_count

def get_largest_fragment(mol):
    """Get the largest fragment from a molecule"""
    frags = Chem.GetMolFrags(mol, asMols=True)
    if not frags:
        return mol
    return max(frags, key=lambda x: x.GetNumAtoms())

def generate_fragments(mol):
    frag_list = rdMMPA.FragmentMol(mol)
    flat_frags = [x for x in itertools.chain(*frag_list) if x]
    flat_frags = [get_largest_fragment(x) for x in flat_frags]
    
    num_atoms = mol.GetNumAtoms()
    flat_frags = [
        x for x in flat_frags
        if x.GetNumAtoms() / num_atoms > 0.67
    ]
    
    flat_frags = [cleanup_fragment(x) for x in flat_frags]
    
    frag_smiles = [
        [Chem.MolToSmiles(x), x.GetNumAtoms(), y]
        for (x, y) in flat_frags
    ]
    
    frag_smiles.append([Chem.MolToSmiles(mol), mol.GetNumAtoms(), 1])
    
    df = pd.DataFrame(
        frag_smiles,
        columns=["Core", "NumAtoms", "NumRgroups"]
    ).drop_duplicates("Core")
    
    return df

# =========================================================
# MMP ANALYSIS WITH PROGRESS TRACKER - CORRECTED
# =========================================================
def run_mmp_with_progress(processed_df, min_occ):
    total_steps = len(processed_df)
    progress_bar = st.progress(0)
    status = st.empty()
    
    row_list = []
    
    # -------- STEP 1: Fragment molecules --------
    status.text("🔬 Fragmenting molecules and assigning R-groups...")
    for i, (smiles, name, activity, mol) in enumerate(
        processed_df[["SMILES", "Name", "Activity", "mol"]].values
    ):
        frag_df = generate_fragments(mol)
        
        for core in frag_df.Core:
            core_mol = Chem.MolFromSmiles(core)
            if core_mol is None:
                continue
            
            try:
                # CORRECTED R-group decomposition
                res, _ = rdRGroupDecomposition.RGroupDecompose([core_mol], [mol])
                if not res:
                    continue
                    
                rgroups = []
                for key, value in res[0].items():
                    if key.startswith('R'):
                        rgroups.append(Chem.MolToSmiles(value))
                
                if not rgroups:
                    continue
                
                row_list.append([
                    smiles, core, rgroups[0], name, activity
                ])
            except:
                continue
        
        progress_bar.progress((i + 1) / total_steps)
    
    if not row_list:
        return None
    
    row_df = pd.DataFrame(
        row_list,
        columns=["SMILES", "Core", "R_group", "Name", "Activity"]
    )
    
    # -------- STEP 2: Pairwise comparison --------
    status.text("🧮 Generating matched molecular pairs...")
    delta_list = []
    groups = list(row_df.groupby("Core"))
    
    for i, (_, v) in enumerate(groups):
        if len(v) > 1:  # Changed from 2 to 1 to allow pairs
            for a, b in combinations(range(len(v)), 2):
                ra, rb = v.iloc[a], v.iloc[b]
                if ra.SMILES == rb.SMILES:
                    continue
                
                ra, rb = sorted([ra, rb], key=lambda x: x.SMILES)
                
                delta = float(rb.Activity) - float(ra.Activity)
                transform = (
                    f"{ra.R_group.replace('*','*-')}"
                    f">>"
                    f"{rb.R_group.replace('*','*-')}"
                )
                
                delta_list.append(
                    list(ra.values) +
                    list(rb.values) +
                    [transform, delta]
                )
        
        progress_bar.progress((i + 1) / len(groups))
    
    if not delta_list:
        return None
    
    delta_df = pd.DataFrame(
        delta_list,
        columns=[
            "SMILES_1", "Core_1", "R_group_1", "Name_1", "Activity_1",
            "SMILES_2", "Core_2", "R_group_2", "Name_2", "Activity_2",
            "Transform", "Delta"
        ]
    )
    
    # -------- STEP 3: Aggregate transforms --------
    status.text("📊 Aggregating transforms and statistics...")
    mmp_list = []
    
    for transform, v in delta_df.groupby("Transform"):
        if len(v) >= min_occ:
            mmp_list.append({
                "Transform": transform,
                "Count": len(v),
                "mean_delta": v.Delta.mean(),
                "std_delta": v.Delta.std(),
                "min_delta": v.Delta.min(),
                "max_delta": v.Delta.max()
            })
    
    mmp_df = pd.DataFrame(mmp_list)
    
    # Remove Deltas from mmp_df as it's now stored in individual stats
    status.text("✅ MMP analysis completed")
    progress_bar.progress(1.0)
    
    return row_df, delta_df, mmp_df

# =========================================================
# Main execution remains the same...
# =========================================================

# The Streamlit UI code remains mostly the same, just update the processing:
if uploaded:
    df = pd.read_csv(uploaded)
    
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())
    
    cols = df.columns.tolist()
    smiles_col = st.selectbox("SMILES column", cols)
    activity_col = st.selectbox("Activity column", cols)
    name_col = st.selectbox("Name column (optional)", ["None"] + cols)
    
    if run_button:
        with st.spinner("Preparing molecules..."):
            proc_df = df.copy()
            proc_df["mol"] = proc_df[smiles_col].apply(Chem.MolFromSmiles)
            proc_df = proc_df[proc_df.mol.notnull()]
            proc_df["mol"] = proc_df.mol.apply(get_largest_fragment)  # Use corrected function
            
            proc_df = proc_df.rename(columns={
                smiles_col: "SMILES",
                activity_col: "Activity"
            })
            
            if name_col != "None":
                proc_df = proc_df.rename(columns={name_col: "Name"})
            else:
                proc_df["Name"] = [
                    f"Mol_{i+1}" for i in range(len(proc_df))
                ]
            
            # Convert activity to float
            proc_df["Activity"] = pd.to_numeric(proc_df["Activity"], errors='coerce')
            proc_df = proc_df.dropna(subset=["Activity"])
        
        result = run_mmp_with_progress(proc_df, min_occ)
        if result is None:
            st.error("No valid MMPs found.")
        else:
            row_df, delta_df, mmp_df = result

            st.success(f"✅ Found {len(mmp_df)} MMP transforms")

            st.subheader("🔬 MMP Summary")
            st.dataframe(
                mmp_df[
                    ["Transform", "Count",
                     "mean_delta", "std_delta",
                     "min_delta", "max_delta"]
                ].sort_values("mean_delta"),
                use_container_width=True
            )

            st.subheader("🧪 All Matched Pairs")
            st.dataframe(delta_df, use_container_width=True)

            st.subheader("🧩 Fragment Table")
            st.dataframe(row_df, use_container_width=True)

            st.subheader("📥 Download Results")
            st.download_button(
                "Download MMP Summary",
                mmp_df.to_csv(index=False),
                "mmp_summary.csv"
            )
            st.download_button(
                "Download All Pairs",
                delta_df.to_csv(index=False),
                "mmp_pairs.csv"
            )

else:
    st.info("👈 Upload a CSV file to begin")

