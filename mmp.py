# -*- coding: utf-8 -*-

import streamlit as st
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
# Helper functions
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
# Safe wrappers
# -----------------------------

def safe_reaction(smarts):
    try:
        return AllChem.ReactionFromSmarts(smarts.replace("*-","*"), useSmiles=True)
    except:
        return None

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("⚙️ Settings")

min_transform_occurrence = st.sidebar.slider(
    "Minimum MMP Occurrence", 2, 20, 5
)

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV (SMILES, Name, pIC50)", type=["csv"]
)

# -----------------------------
# Main
# -----------------------------
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    df["mol"] = df.SMILES.apply(Chem.MolFromSmiles)
    df["mol"] = df.mol.apply(uru.get_largest_fragment)

    # Fragmentation
    row_list = []
    for smiles, name, pIC50, mol in df.values:
        frag_list = FragmentMol(mol, maxCuts=1)
        for _, frag_mol in frag_list:
            pair_list = sort_fragments(frag_mol)
            row_list.append(
                [smiles] +
                [Chem.MolToSmiles(x) for x in pair_list] +
                [name, pIC50]
            )

    row_df = pd.DataFrame(
        row_list,
        columns=["SMILES","Core","R_group","Name","pIC50"]
    )

    # Δ calculation
    delta_list = []
    for k, v in row_df.groupby("Core"):
        if len(v) > 2:
            for a, b in combinations(range(len(v)), 2):
                ra, rb = v.iloc[a], v.iloc[b]
                if ra.SMILES == rb.SMILES:
                    continue
                ra, rb = sorted([ra, rb], key=lambda x: x.SMILES)
                delta = rb.pIC50 - ra.pIC50
                delta_list.append(
                    list(ra.values) +
                    list(rb.values) +
                    [f"{ra.R_group.replace('*','*-')}>>{rb.R_group.replace('*','*-')}", delta]
                )

    delta_df = pd.DataFrame(delta_list, columns=[
        "SMILES_1","Core_1","R_group_1","Name_1","pIC50_1",
        "SMILES_2","Core_2","Rgroup_2","Name_2","pIC50_2",
        "Transform","Delta"
    ])

    # Aggregate MMPs
    mmp_rows = []
    for k, v in delta_df.groupby("Transform"):
        if len(v) > min_transform_occurrence:
            mmp_rows.append([k, len(v), np.array(v.Delta)])

    mmp_df = pd.DataFrame(mmp_rows, columns=["Transform","Count","Deltas"])
    mmp_df["mean_delta"] = mmp_df.Deltas.apply(np.mean)
    mmp_df["rxn"] = mmp_df.Transform.apply(safe_reaction)

    # ---- CRITICAL FIX ----
    # Remove NumPy arrays BEFORE Streamlit rendering
    mmp_df["MMP Transform"] = mmp_df["rxn"].apply(
        lambda x: rxn_to_base64_image(x) if x else ""
    )
    mmp_df["Delta Distribution"] = mmp_df["Deltas"].apply(stripplot_base64_image)

    # Drop problematic columns
    mmp_df = mmp_df.drop(columns=["Deltas","rxn"])

    mmp_df.sort_values("mean_delta", inplace=True)

    st.markdown(
        mmp_df[["MMP Transform","Count","mean_delta","Delta Distribution"]]
        .round(2)
        .to_html(escape=False),
        unsafe_allow_html=True
    )

else:
    st.info("Upload a CSV file to start.")

