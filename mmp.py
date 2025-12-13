# -*- coding: utf-8 -*-
# NOTE: This file is IDENTICAL to your original code
# EXCEPT the fragmentation strategy, which now uses RDKit rdMMPA (true MMP logic)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from operator import itemgetter
from itertools import combinations
import sys
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="MMP Analysis Tool",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- RDKit -----------------
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.rdMMPA import FragmentMol as RDKitFragmentMol
    RDKIT_AVAILABLE = True
except Exception as e:
    RDKIT_AVAILABLE = False

# ---------------- Fragmentation (ONLY CHANGE) -----------------

def fragment_molecule_mmp(mol):
    """True RDKit MMP single-cut fragmentation"""
    results = []
    try:
        frags = RDKitFragmentMol(mol, maxCuts=1, resultsAsMols=True)
        for _, frag_mols in frags:
            if frag_mols is None or len(frag_mols) < 2:
                continue
            frag_mols = sorted(frag_mols, key=lambda x: x.GetNumAtoms(), reverse=True)
            core, rgroup = frag_mols[0], frag_mols[1]
            for a in core.GetAtoms():
                a.SetAtomMapNum(0)
            for a in rgroup.GetAtoms():
                a.SetAtomMapNum(0)
            results.append((
                Chem.MolToSmiles(core, canonical=True),
                Chem.MolToSmiles(rgroup, canonical=True)
            ))
    except Exception:
        pass
    return results

# ---------------- Data Loading -----------------

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df['mol'] = df.SMILES.apply(Chem.MolFromSmiles)
    df = df[df.mol.notna()].copy()
    return df

# ---------------- MMP Analysis -----------------

def perform_mmp_analysis(df, min_occurrence):
    row_list = []

    for idx, row in df.iterrows():
        mol = row.mol
        if mol is None:
            continue
        pairs = fragment_molecule_mmp(mol)
        for core, rgroup in pairs:
            row_list.append([
                row.SMILES,
                core,
                rgroup,
                row.get('Name', f'CMPD_{idx}'),
                row.pIC50
            ])

    row_df = pd.DataFrame(row_list, columns=["SMILES", "Core", "R_group", "Name", "pIC50"])

    delta_list = []
    for core, v in row_df.groupby("Core"):
        if len(v) > 2:
            for a, b in combinations(range(len(v)), 2):
                ra, rb = v.iloc[a], v.iloc[b]
                if ra.SMILES == rb.SMILES:
                    continue
                ra, rb = sorted([ra, rb], key=lambda x: x.SMILES)
                delta_list.append([
                    ra.SMILES, ra.Core, ra.R_group, ra.Name, ra.pIC50,
                    rb.SMILES, rb.Core, rb.R_group, rb.Name, rb.pIC50,
                    f"{ra.R_group}>>{rb.R_group}",
                    rb.pIC50 - ra.pIC50
                ])

    delta_df = pd.DataFrame(delta_list, columns=[
        "SMILES_1", "Core_1", "R_group_1", "Name_1", "pIC50_1",
        "SMILES_2", "Core_2", "R_group_2", "Name_2", "pIC50_2",
        "Transform", "Delta"
    ])

    mmp_list = []
    for k, v in delta_df.groupby("Transform"):
        if len(v) >= min_occurrence:
            mmp_list.append([k, len(v), v.Delta.values])

    mmp_df = pd.DataFrame(mmp_list, columns=["Transform", "Count", "Deltas"])
    if not mmp_df.empty:
        mmp_df['mean_delta'] = mmp_df.Deltas.apply(np.mean)

    return delta_df, mmp_df

# ---------------- UI -----------------

st.title("🧪 Matched Molecular Pair (MMP) Analysis Tool")

uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
min_occurrence = st.slider("Minimum transform occurrence", 1, 20, 5)

if uploaded_file and RDKIT_AVAILABLE:
    df = load_data(uploaded_file)
    delta_df, mmp_df = perform_mmp_analysis(df, min_occurrence)

    st.success("Analysis complete")
    st.metric("Total Pairs", len(delta_df))

    if mmp_df is not None and not mmp_df.empty:
        st.metric("Unique Transforms", len(mmp_df))
        st.dataframe(mmp_df.sort_values("mean_delta", ascending=False))

        st.download_button(
            "Download MMP Results",
            mmp_df.to_csv(index=False),
            file_name="mmp_results.csv"
        )
else:
    st.info("Upload a CSV file with SMILES and pIC50 columns")
