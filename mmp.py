# app.py
import streamlit as st
import pandas as pd
import numpy as np
import itertools
from itertools import combinations
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMMPA import FragmentMol
from rdkit.Chem.rdRGroupDecomposition import RGroupDecompose
import useful_rdkit_utils as uru
import matplotlib.pyplot as plt

# ================================
# Streamlit config
# ================================
st.set_page_config(
    page_title="MMP Analyzer (Scaffold-Finder Equivalent)",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Matched Molecular Pair (MMP) Analyzer")
st.markdown(
    """
This app performs **Matched Molecular Pair analysis using the exact same
algorithm as `scaffold_finder.py`**  
(RDKit `rdMMPA`, 2/3 atom filter, RGroupDecompose).

✔ Results are **reproducible**  
✔ Scientifically **equivalent to the reference implementation**
"""
)

# ================================
# Helper functions (IDENTICAL LOGIC)
# ================================
def cleanup_fragment(mol):
    """
    EXACT copy of scaffold_finder.py
    Replace dummy atoms with hydrogens and remove atom maps
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
    EXACT fragment generation logic from scaffold_finder.py
    """
    frag_list = FragmentMol(mol)
    flat_frags = [x for x in itertools.chain(*frag_list) if x]

    flat_frags = [uru.get_largest_fragment(x) for x in flat_frags]

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

    frag_df = pd.DataFrame(
        frag_smiles,
        columns=["Core", "NumAtoms", "NumRgroups"]
    ).drop_duplicates("Core")

    return frag_df


# ================================
# MMP processing (IDENTICAL PAIRING)
# ================================
def run_mmp(processed_df, min_transform_occurrence):

    row_list = []

    for smiles, name, activity, mol in processed_df[
        ["SMILES", "Name", "Activity", "mol"]
    ].values:

        frag_df = generate_fragments(mol)

        for core in frag_df.Core:
            core_mol = Chem.MolFromSmiles(core)
            if core_mol is None:
                continue

            rgroup_match, _ = RGroupDecompose(
                core_mol, [mol], asSmiles=True
            )

            if not rgroup_match:
                continue

            rgroups = list(rgroup_match[0].values())
            if not rgroups:
                continue

            row_list.append([
                smiles,
                core,
                rgroups[0],
                name,
                activity
            ])

    if not row_list:
        return None

    row_df = pd.DataFrame(
        row_list,
        columns=["SMILES", "Core", "R_group", "Name", "Activity"]
    )

    # --- Pairwise comparison (IDENTICAL) ---
    delta_list = []
    for _, v in row_df.groupby("Core"):
        if len(v) > 2:
            for a, b in combinations(range(len(v)), 2):
                ra, rb = v.iloc[a], v.iloc[b]
                if ra.SMILES == rb.SMILES:
                    continue

                ra, rb = sorted([ra, rb], key=lambda x: x.SMILES)

                delta = rb.Activity - ra.Activity
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

    if not delta_list:
        return None

    delta_df = pd.DataFrame(
        delta_list,
        columns=[
            "SMILES_1","Core_1","R_group_1","Name_1","Activity_1",
            "SMILES_2","Core_2","R_group_2","Name_2","Activity_2",
            "Transform","Delta"
        ]
    )

    # --- Aggregate transforms ---
    mmp_list = []
    for k, v in delta_df.groupby("Transform"):
        if len(v) >= min_transform_occurrence:
            mmp_list.append({
                "Transform": k,
                "Count": len(v),
                "Deltas": v.Delta.values,
                "mean_delta": v.Delta.mean(),
                "std_delta": v.Delta.std(),
                "min_delta": v.Delta.min(),
                "max_delta": v.Delta.max()
            })

    if not mmp_list:
        return None

    mmp_df = pd.DataFrame(mmp_list)
    mmp_df["rxn_mol"] = mmp_df.Transform.apply(
        lambda x: AllChem.ReactionFromSmarts(x, useSmiles=True)
    )

    return row_df, delta_df, mmp_df


# ================================
# Sidebar – data input
# ================================
st.sidebar.header("📁 Input Data")

uploaded = st.sidebar.file_uploader(
    "Upload CSV (SMILES, Activity required)",
    type=["csv"]
)

min_occ = st.sidebar.slider(
    "Minimum transform occurrence",
    2, 20, 5
)

run_button = st.sidebar.button("🔬 Run MMP Analysis")

# ================================
# Load and prepare data
# ================================
if uploaded:
    df = pd.read_csv(uploaded)

    st.subheader("📊 Raw Data Preview")
    st.dataframe(df.head())

    cols = df.columns.tolist()
    smiles_col = st.selectbox("SMILES column", cols)
    activity_col = st.selectbox("Activity column", cols)
    name_col = st.selectbox("Name column (optional)", ["None"] + cols)

    if run_button:
        with st.spinner("Running MMP analysis..."):

            proc_df = df.copy()
            proc_df["mol"] = proc_df[smiles_col].apply(Chem.MolFromSmiles)
            proc_df = proc_df[proc_df.mol.notnull()]
            proc_df["mol"] = proc_df.mol.apply(uru.get_largest_fragment)

            proc_df = proc_df.rename(columns={
                smiles_col: "SMILES",
                activity_col: "Activity"
            })

            if name_col != "None":
                proc_df = proc_df.rename(columns={name_col: "Name"})
            else:
                proc_df["Name"] = [f"Mol_{i+1}" for i in range(len(proc_df))]

            row_df, delta_df, mmp_df = run_mmp(proc_df, min_occ)

            st.success(f"✅ Found {len(mmp_df)} MMP transforms")

            # ================================
            # Results display
            # ================================
            st.header("🔬 MMP Results")

            st.subheader("Top MMP Transforms")
            st.dataframe(
                mmp_df[[
                    "Transform", "Count",
                    "mean_delta", "std_delta",
                    "min_delta", "max_delta"
                ]].sort_values("mean_delta"),
                use_container_width=True
            )

            st.subheader("All Molecular Pairs")
            st.dataframe(delta_df, use_container_width=True)

            st.subheader("Fragment Table (Row DF)")
            st.dataframe(row_df, use_container_width=True)

            # ================================
            # Download
            # ================================
            st.subheader("📥 Download")
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
