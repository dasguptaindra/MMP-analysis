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

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="MMP Analysis Tool",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Matched Molecular Pair (MMP) Analysis Tool")

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
with st.sidebar:
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    min_occurrence = st.slider(
        "Minimum transform occurrences",
        min_value=1,
        max_value=20,
        value=2,
        help="Recommended: 1–2 for small datasets"
    )

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)

    if not {"SMILES", "pIC50"}.issubset(df.columns):
        st.error("CSV must contain SMILES and pIC50 columns")
        return None

    df["mol"] = df["SMILES"].apply(
        lambda x: Chem.MolFromSmiles(str(x))
    )

    df = df[df.mol.notna()].copy()
    return df

# ---------------------------------------------------
# MMP ANALYSIS
# ---------------------------------------------------
def perform_mmp_analysis(df, min_occurrence):

    # ---------- STEP 1: FRAGMENTATION ----------
    row_list = []

    for idx, row in df.iterrows():
        mol = row.mol
        smiles = row.SMILES
        pIC50 = row.pIC50
        name = row.get("Name", f"CMPD_{idx}")

        frags = FragmentMol(
            mol,
            maxCuts=1,
            resultsAsMols=True
        )

        for core, chains in frags:

            if core is None:
                continue
            if not chains or chains[0] is None:
                continue

            row_list.append([
                smiles,
                Chem.MolToSmiles(core),
                Chem.MolToSmiles(chains[0]),
                name,
                pIC50
            ])

    row_df = pd.DataFrame(
        row_list,
        columns=["SMILES", "Core", "R_group", "Name", "pIC50"]
    )

    if row_df.empty:
        return None, None, None

    # ---------- STEP 2: DELTA GENERATION ----------
    delta_list = []

    for core, v in row_df.groupby("Core"):
        if len(v) > 2:
            for a, b in combinations(range(len(v)), 2):

                r1 = v.iloc[a]
                r2 = v.iloc[b]

                if r1.SMILES == r2.SMILES:
                    continue

                r1, r2 = sorted([r1, r2], key=lambda x: x.SMILES)

                delta_list.append(
                    list(r1.values) +
                    list(r2.values) +
                    [
                        f"{r1.R_group.replace('*','*-')}>>"
                        f"{r2.R_group.replace('*','*-')}",
                        r2.pIC50 - r1.pIC50
                    ]
                )

    delta_cols = (
        list(row_df.columns) +
        list(row_df.columns) +
        ["Transform", "Delta"]
    )

    delta_df = pd.DataFrame(delta_list, columns=delta_cols)

    if delta_df.empty:
        return row_df, delta_df, None

    # ---------- STEP 3: TRANSFORM AGGREGATION ----------
    mmp_df = (
        delta_df
        .groupby("Transform")
        .agg(
            Count=("Delta", "size"),
            mean_delta=("Delta", "mean"),
            std_delta=("Delta", "std")
        )
        .reset_index()
    )

    mmp_df["std_delta"] = mmp_df["std_delta"].fillna(0.0)

    # APPLY OCCURRENCE FILTER *AFTER* COMPUTATION
    filtered_mmp_df = mmp_df[mmp_df.Count >= min_occurrence]

    return row_df, delta_df, filtered_mmp_df

# ---------------------------------------------------
# MAIN APP
# ---------------------------------------------------
if uploaded_file:

    df = load_data(uploaded_file)

    if df is not None:
        st.success(f"Loaded {len(df)} molecules")

        row_df, delta_df, mmp_df = perform_mmp_analysis(
            df,
            min_occurrence
        )

        # ---------------- METRICS ----------------
        col1, col2, col3 = st.columns(3)
        col1.metric("Fragments generated", len(row_df))
        col2.metric("MMP pairs", len(delta_df))
        col3.metric("Unique transforms", delta_df.Transform.nunique())

        # ---------------- TRANSFORMS ----------------
        st.subheader("📈 Transform Summary")

        if mmp_df is not None and not mmp_df.empty:
            st.dataframe(
                mmp_df.sort_values("mean_delta", ascending=False)
                .round(3)
            )
        else:
            st.warning(
                "No transforms passed the occurrence threshold.\n"
                "Try lowering the minimum occurrence value."
            )

        # ---------------- RAW DATA ----------------
        with st.expander("🔍 View all MMP pairs"):
            st.dataframe(delta_df)

else:
    st.info("⬅ Upload a CSV file to begin")
