# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdMMPA import FragmentMol
from rdkit.Chem import AllChem
from operator import itemgetter
from itertools import combinations
from tqdm import tqdm
import useful_rdkit_utils as uru
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from rdkit.Chem.Draw import rdMolDraw2D

# -----------------------------
# Streamlit config
# -----------------------------
st.set_page_config(
    page_title="MMP Analysis Tool",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Matched Molecular Pair (MMP) Analysis")

# -----------------------------
# Helper functions (UNCHANGED LOGIC)
# -----------------------------

def remove_map_nums(mol):
    for atm in mol.GetAtoms():
        atm.SetAtomMapNum(0)

def sort_fragments(mol):
    frag_list = list(Chem.GetMolFrags(mol, asMols=True))
    [remove_map_nums(x) for x in frag_list]
    frag_num_atoms_list = [(x.GetNumAtoms(), x) for x in frag_list]
    frag_num_atoms_list.sort(key=itemgetter(0), reverse=True)
    return [x[1] for x in frag_num_atoms_list]

def rxn_to_base64_image(rxn, size=(300,150)):
    if rxn is None:
        return ""
    drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    drawer.DrawReaction(rxn)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    return f'<img src="data:image/svg+xml;base64,{b64}"/>'

def stripplot_base64_image(delta_values):
    fig, ax = plt.subplots(figsize=(3,2))
    sns.stripplot(y=delta_values, ax=ax)
    ax.set_ylabel("ΔpIC50")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)

    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f'<img src="data:image/png;base64,{b64}"/>'

# -----------------------------
# Sidebar controls
# -----------------------------

st.sidebar.header("⚙️ Settings")
min_transform_occurrence = st.sidebar.slider(
    "Minimum MMP Occurrence",
    min_value=2,
    max_value=20,
    value=5
)

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV (SMILES, Name, pIC50)",
    type=["csv"]
)

# -----------------------------
# Main workflow
# -----------------------------

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Input Data")
    st.dataframe(df.head())

    with st.spinner("Generating RDKit molecules..."):
        df['mol'] = df.SMILES.apply(Chem.MolFromSmiles)
        df['mol'] = df.mol.apply(uru.get_largest_fragment)

    # -----------------------------
    # Fragment generation
    # -----------------------------
    st.subheader("🔬 Fragmenting molecules")

    row_list = []
    progress = st.progress(0)

    for i, (smiles, name, pIC50, mol) in enumerate(tqdm(df.values)):
        frag_list = FragmentMol(mol, maxCuts=1)
        for _, frag_mol in frag_list:
            pair_list = sort_fragments(frag_mol)
            tmp = (
                [smiles] +
                [Chem.MolToSmiles(x) for x in pair_list] +
                [name, pIC50]
            )
            row_list.append(tmp)

        progress.progress((i + 1) / len(df))

    row_df = pd.DataFrame(
        row_list,
        columns=["SMILES", "Core", "R_group", "Name", "pIC50"]
    )

    st.subheader("🧬 Core–R Group Table")
    st.dataframe(row_df.head(20))

    # -----------------------------
    # MMP delta calculation
    # -----------------------------
    st.subheader("📐 Calculating ΔpIC50")

    delta_list = []
    progress = st.progress(0)

    grouped = list(row_df.groupby("Core"))

    for i, (k, v) in enumerate(tqdm(grouped)):
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
        progress.progress((i + 1) / len(grouped))

    cols = [
        "SMILES_1","Core_1","R_group_1","Name_1","pIC50_1",
        "SMILES_2","Core_2","Rgroup_2","Name_2","pIC50_2",
        "Transform","Delta"
    ]

    delta_df = pd.DataFrame(delta_list, columns=cols)

    st.subheader("📊 Delta Table")
    st.dataframe(delta_df.head(20))

    # -----------------------------
    # Aggregate MMPs
    # -----------------------------
    st.subheader("🔁 Aggregating MMP Transforms")

    mmp_list = []
    for k, v in delta_df.groupby("Transform"):
        if len(v) > min_transform_occurrence:
            mmp_list.append([k, len(v), v.Delta.values])

    mmp_df = pd.DataFrame(
        mmp_list,
        columns=["Transform","Count","Deltas"]
    )

    mmp_df["mean_delta"] = mmp_df.Deltas.apply(np.mean)
    mmp_df["rxn_mol"] = mmp_df.Transform.apply(
        lambda x: AllChem.ReactionFromSmarts(
            x.replace("*-","*"),
            useSmiles=True
        )
    )

    with st.spinner("Rendering MMP visuals..."):
        mmp_df["MMP Transform"] = mmp_df.rxn_mol.apply(rxn_to_base64_image)
        mmp_df["Delta Distribution"] = mmp_df.Deltas.apply(stripplot_base64_image)

    mmp_df.sort_values("mean_delta", ascending=True, inplace=True)

    # -----------------------------
    # Display results
    # -----------------------------
    st.subheader("🏆 Final MMP Results")

    st.markdown(
        mmp_df[
            ["MMP Transform","Count","mean_delta","Delta Distribution"]
        ].round(2).to_html(escape=False),
        unsafe_allow_html=True
    )

    # -----------------------------
    # Download
    # -----------------------------
    st.subheader("⬇️ Download Results")

    csv = mmp_df.drop(
        columns=["rxn_mol","MMP Transform","Delta Distribution"]
    ).to_csv(index=False)

    st.download_button(
        "Download MMP Table (CSV)",
        csv,
        "mmp_results.csv",
        "text/csv"
    )

else:
    st.info("👈 Upload a CSV file to start MMP analysis.")
