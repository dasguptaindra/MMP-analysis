# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")

# RDKit
from rdkit import Chem
from rdkit.Chem.rdMMPA import FragmentMol

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="MMP Analysis Tool",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Matched Molecular Pair (MMP) Analysis Tool")

# -------------------- SIDEBAR --------------------
with st.sidebar:
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    min_occurrence = st.slider(
        "Minimum transform occurrences",
        1, 20, 5
    )

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)

    if not {"SMILES", "pIC50"}.issubset(df.columns):
        st.error("CSV must contain SMILES and pIC50 columns")
        return None

    mols = []
    for s in df.SMILES:
        mols.append(Chem.MolFromSmiles(str(s)))

    df["mol"] = mols
    df = df[df.mol.notna()].copy()

    return df

# -------------------- MMP ANALYSIS --------------------
def perform_mmp_analysis(df, min_occurrence):

    # ---------- STEP 1: FRAGMENTATION ----------
    row_list = []

    for i, row in df.iterrows():
        mol = row.mol
        smiles = row.SMILES
        pIC50 = row.pIC50
        name = row.get("Name", f"CMPD_{i}")

        frags = FragmentMol(
            mol,
            maxCuts=1,
            resultsAsMols=True
        )

        for core, chains in frags:

            # SAFETY CHECKS
            if core is None:
                continue
            if not chains or chains[0] is None:
                continue

            core_smiles = Chem.MolToSmiles(core)
            r_smiles = Chem.MolToSmiles(chains[0])

            row_list.append([
                smiles,
                core_smiles,
                r_smiles,
                name,
                pIC50
            ])

    row_df = pd.DataFrame(
        row_list,
        columns=["SMILES", "Core", "R_group", "Name", "pIC50"]
    )

    if row_df.empty:
        return None, None

    # ---------- STEP 2: EXACT ORIGINAL DELTA LOGIC ----------
    delta_list = []

    for k, v in row_df.groupby("Core"):
        if len(v) > 2:
            for a, b in combinations(range(len(v)), 2):

                reagent_a = v.iloc[a]
                reagent_b = v.iloc[b]

                if reagent_a.SMILES == reagent_b.SMILES:
                    continue

                reagent_a, reagent_b = sorted(
                    [reagent_a, reagent_b],
                    key=lambda x: x.SMILES
                )

                delta = reagent_b.pIC50 - reagent_a.pIC50

                delta_list.append(
                    list(reagent_a.values) +
                    list(reagent_b.values) +
                    [
                        f"{reagent_a.R_group.replace('*','*-')}>>"
                        f"{reagent_b.R_group.replace('*','*-')}",
                        delta
                    ]
                )

    delta_cols = (
        list(row_df.columns) +
        list(row_df.columns) +
        ["Transform", "Delta"]
    )

    delta_df = pd.DataFrame(delta_list, columns=delta_cols)

    # ---------- STEP 3: TRANSFORM AGGREGATION ----------
    mmp_list = []

    for k, v in delta_df.groupby("Transform"):
        if len(v) >= min_occurrence:
            mmp_list.append([
                k,
                len(v),
                v.Delta.values
            ])

    if not mmp_list:
        return delta_df, None

    mmp_df = pd.DataFrame(
        mmp_list,
        columns=["Transform", "Count", "Deltas"]
    )

    mmp_df["mean_delta"] = mmp_df.Deltas.apply(np.mean)

    return delta_df, mmp_df

# -------------------- MAIN --------------------
if uploaded_file:
    df = load_data(uploaded_file)

    if df is not None:
        st.success(f"Loaded {len(df)} molecules")

        delta_df, mmp_df = perform_mmp_analysis(
            df,
            min_occurrence
        )

        if delta_df is not None:
            st.metric("Total MMP pairs", len(delta_df))
            st.subheader("All MMP Pairs")
            st.dataframe(delta_df)

        if mmp_df is not None:
            st.metric("Unique transforms", len(mmp_df))
            st.subheader("Top Transforms")
            st.dataframe(
                mmp_df
                .sort_values("mean_delta", ascending=False)
                [["Transform", "Count", "mean_delta"]]
                .round(3)
            )
        else:
            st.warning("No transforms passed the occurrence threshold")

else:
    st.info("⬅ Upload a CSV file to start")
