# -*- coding: utf-8 -*-

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdMMPA import FragmentMol
from rdkit.Chem import AllChem
from operator import itemgetter
from itertools import combinations
import useful_rdkit_utils as uru
import matplotlib.pyplot as plt
import seaborn as sns
import io, base64
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

def rxn_to_base64_image(rxn):
    drawer = rdMolDraw2D.MolDraw2DSVG(300,150)
    drawer.DrawReaction(rxn)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    b64 = base64.b64encode(svg.encode()).decode()
    return f'<img src="data:image/svg+xml;base64,{b64}"/>'

def stripplot_base64_image(deltas):
    fig, ax = plt.subplots(figsize=(3,2))
    sns.stripplot(y=deltas, ax=ax)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return f'<img src="data:image/png;base64,{b64}"/>'

# -----------------------------
# Sidebar
# -----------------------------
min_transform_occurrence = st.sidebar.slider(
    "Minimum MMP Occurrence", 2, 20, 5
)

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV (SMILES, Name, pIC50)", type=["csv"]
)

# -----------------------------
# Main
# -----------------------------
if uploaded_file:

    df = pd.read_csv(uploaded_file)
    st.subheader("Input Data")
    st.dataframe(df.head())

    df["mol"] = df.SMILES.apply(Chem.MolFromSmiles)
    df["mol"] = df.mol.apply(uru.get_largest_fragment)

    # Fragmentation
    row_list = []
    for smiles, name, pIC50, mol in df.values:
        for _, frag_mol in FragmentMol(mol, maxCuts=1):
            pair = sort_fragments(frag_mol)
            row_list.append(
                [smiles] +
                [Chem.MolToSmiles(x) for x in pair] +
                [name, pIC50]
            )

    row_df = pd.DataFrame(
        row_list,
        columns=["SMILES","Core","R_group","Name","pIC50"]
    )

    # Delta calculation
    delta_rows = []
    for _, v in row_df.groupby("Core"):
        if len(v) > 2:
            for a,b in combinations(range(len(v)),2):
                ra, rb = v.iloc[a], v.iloc[b]
                if ra.SMILES == rb.SMILES:
                    continue
                ra, rb = sorted([ra, rb], key=lambda x: x.SMILES)
                delta_rows.append(
                    list(ra.values) +
                    list(rb.values) +
                    [
                        f"{ra.R_group.replace('*','*-')}>>{rb.R_group.replace('*','*-')}",
                        rb.pIC50 - ra.pIC50
                    ]
                )

    delta_df = pd.DataFrame(delta_rows, columns=[
        "SMILES_1","Core_1","R_group_1","Name_1","pIC50_1",
        "SMILES_2","Core_2","Rgroup_2","Name_2","pIC50_2",
        "Transform","Delta"
    ])

    # Aggregate MMPs
    mmp_rows = []
    for k, v in delta_df.groupby("Transform"):
        if len(v) > min_transform_occurrence:
            mmp_rows.append([k, len(v), v.Delta.values])

    mmp_df = pd.DataFrame(mmp_rows, columns=["Transform","Count","Deltas"])
    mmp_df["mean_delta"] = mmp_df.Deltas.apply(np.mean)
    mmp_df["rxn"] = mmp_df.Transform.apply(
        lambda x: AllChem.ReactionFromSmarts(x.replace("*-","*"), useSmiles=True)
    )

    # Build HTML columns
    mmp_df["MMP Transform"] = mmp_df["rxn"].apply(rxn_to_base64_image)
    mmp_df["Delta Distribution"] = mmp_df["Deltas"].apply(stripplot_base64_image)

    # FINAL DISPLAY TABLE (HTML ONLY)
    html_table = (
        mmp_df[["MMP Transform","Count","mean_delta","Delta Distribution"]]
        .round(2)
        .to_html(escape=False, index=False)
    )

    st.subheader("Final MMP Results")

    # 🚀 THIS IS THE KEY LINE (NO STREAMLIT SERIALIZATION)
    components.html(
        html_table,
        height=900,
        scrolling=True
    )

else:
    st.info("Upload a CSV file to start analysis.")
