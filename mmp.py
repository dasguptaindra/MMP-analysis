# streamlit_mmp_app.py
import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdMMPA import FragmentMol
import matplotlib.pyplot as plt
import seaborn as sns
import io
from operator import itemgetter
from itertools import combinations
import base64
import useful_rdkit_utils as uru  # you used this in original notebook
from functools import partial

# ========== UI setup ==========
st.set_page_config(layout="wide", page_title="MMP Analysis (Streamlit)")
st.title("🔁 Matched Molecular Pair (MMP) Analysis — Streamlit")
st.markdown(
    "Upload a CSV with columns `SMILES`, `Name`, `pIC50`. "
    "This app will find cores + R-groups, build MMP transforms, and summarize frequent transforms."
)

# ========== Helper functions ==========
def remove_map_nums(mol):
    for atm in mol.GetAtoms():
        atm.SetAtomMapNum(0)

def sort_fragments(mol):
    """Return fragments sorted by atom count descending (largest first)."""
    frag_list = list(Chem.GetMolFrags(mol, asMols=True))
    [remove_map_nums(x) for x in frag_list]
    frag_num_atoms_list = [(x.GetNumAtoms(), x) for x in frag_list]
    frag_num_atoms_list.sort(key=itemgetter(0), reverse=True)
    return [x[1] for x in frag_num_atoms_list]

@st.cache_data(show_spinner=False)
def read_and_prepare(df):
    """Given uploaded DataFrame ensure columns and compute RDKit molecules and largest fragment."""
    # Ensure correct column names
    needed = ["SMILES", "Name", "pIC50"]
    if not all(c in df.columns for c in needed):
        # try to interpret columns if uploaded file has no headers
        if df.shape[1] >= 3:
            df = df.copy()
            df.columns = df.columns[:3]  # preserve extra col names if present
            df = df.rename(columns={df.columns[0]: "SMILES", df.columns[1]: "Name", df.columns[2]: "pIC50"})
        else:
            raise ValueError("Input CSV must have at least three columns (SMILES, Name, pIC50)")

    df = df[["SMILES", "Name", "pIC50"]].copy()
    # Convert SMILES -> RDKit mol; standardize SMILES
    df["mol"] = df.SMILES.apply(Chem.MolFromSmiles)
    df = df[~df.mol.isna()].copy()
    df["SMILES"] = df.mol.apply(Chem.MolToSmiles)
    # Keep only largest fragment
    df["mol"] = df.mol.apply(uru.get_largest_fragment)
    return df.reset_index(drop=True)

@st.cache_data(show_spinner=False)
def decompose_all(df):
    """Decompose each molecule into Core + Rgroup (using FragmentMol with maxCuts=1)."""
    row_list = []
    for smiles, name, pIC50, mol in df[["SMILES", "Name", "pIC50", "mol"]].itertuples(index=False):
        # use FragmentMol with maxCuts=1 to generate single cut fragments (core + R)
        frag_list = FragmentMol(mol, maxCuts=1)
        for _, frag_mol in frag_list:
            pair_list = sort_fragments(frag_mol)
            # pair_list typically: [core, rgroups...], but with maxCuts=1 it should give 2 fragments
            tmp_list = [smiles] + [Chem.MolToSmiles(x) for x in pair_list] + [name, pIC50]
            row_list.append(tmp_list)
    # determine column names dynamically: many pair_list items => we only expect Core and R_group (first two)
    # But to be robust, we take the first two fragments only (core, R_group)
    row_df = pd.DataFrame(row_list)
    # Normalize to columns: SMILES, Core, R_group, Name, pIC50
    if row_df.shape[1] >= 5:
        row_df = row_df.iloc[:, :5]
        row_df.columns = ["SMILES", "Core", "R_group", "Name", "pIC50"]
    else:
        raise ValueError("Fragmentation returned unexpected number of fragments per molecule.")
    return row_df

@st.cache_data(show_spinner=False)
def build_delta_df(row_df):
    """Given row_df with Core + R_group produce delta_df (pairs sharing same Core)."""
    delta_list = []
    for core, group in row_df.groupby("Core"):
        v = group.reset_index(drop=True)
        if len(v) > 1:
            for a, b in combinations(range(0, len(v)), 2):
                reagent_a = v.iloc[a]
                reagent_b = v.iloc[b]
                if reagent_a.SMILES == reagent_b.SMILES:
                    continue
                # canonical order by SMILES (makes transform canonical)
                reagent_a, reagent_b = sorted([reagent_a, reagent_b], key=lambda x: x.SMILES)
                delta = reagent_b.pIC50 - reagent_a.pIC50
                transform = f"{reagent_a.R_group.replace('*','*-')}>>{reagent_b.R_group.replace('*','*-')}"
                delta_list.append(list(reagent_a.values) + list(reagent_b.values) + [transform, delta])
    cols = ["SMILES_1","Core_1","R_group_1","Name_1","pIC50_1",
           "SMILES_2","Core_2","R_group_2","Name_2","pIC50_2",
           "Transform","Delta"]
    delta_df = pd.DataFrame(delta_list, columns=cols)
    return delta_df

@st.cache_data(show_spinner=False)
def build_mmp_df(delta_df, min_transform_occurrence=5):
    """Aggregate transforms occurring >= min_transform_occurrence into mmp_df with Deltas, mean_delta, rxn_mol"""
    mmp_list = []
    for transform, v in delta_df.groupby("Transform"):
        if len(v) >= min_transform_occurrence:
            mmp_list.append([transform, len(v), v.Delta.values])
    mmp_df = pd.DataFrame(mmp_list, columns=["Transform","Count","Deltas"])
    if mmp_df.empty:
        return mmp_df
    mmp_df = mmp_df.reset_index(drop=True)
    mmp_df["idx"] = mmp_df.index
    mmp_df["mean_delta"] = mmp_df["Deltas"].apply(lambda arr: np.mean(arr))
    # Some transforms include the '*-' escaped form; convert back to reaction SMARTS and create RDKit reaction
    def make_rxn(transform):
        try:
            smt = transform.replace('*-','*')
            rxn = AllChem.ReactionFromSmarts(smt, useSmiles=True)
            return rxn
        except Exception:
            return None
    mmp_df["rxn_mol"] = mmp_df.Transform.apply(make_rxn)
    # map idx back to delta_df
    transform_dict = dict(mmp_df[["Transform","idx"]].values)
    delta_df["idx"] = delta_df.Transform.map(transform_dict)
    return mmp_df, delta_df

def rxn_image_bytes(rxn, subImgSize=(300,150)):
    """Return PNG bytes for RDKit reaction object (or None)."""
    if rxn is None:
        return None
    try:
        pil = Draw.ReactionToImage(rxn, subImgSize=subImgSize)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        # fallback: draw using rdMolDraw2D
        try:
            drawer = rdMolDraw2D.MolDraw2DCairo(subImgSize[0], subImgSize[1])
            drawer.DrawReaction(rxn)
            drawer.FinishDrawing()
            png = drawer.GetDrawingText()
            return png
        except Exception:
            return None

def plot_delta_strip(dist, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(4,1))
    sns.stripplot(x=dist, jitter=0.2, alpha=0.7)
    ax.axvline(0, ls="--", color="red")
    ax.set_xlim(-5,5)
    ax.set_xlabel("ΔpIC50")
    ax.set_yticks([])
    return ax

# ========== Sidebar: upload & options ==========
st.sidebar.header("Input / Options")
uploaded_file = st.sidebar.file_uploader("Upload SMILES CSV (.csv or .smi)", type=["csv","smi"])
min_occ = st.sidebar.slider("Minimum transform occurrences",  min_value=2, max_value=50, value=5, step=1)
show_top_n = st.sidebar.number_input("Number of top transforms to display", value=5, min_value=1, max_value=50, step=1)

if uploaded_file is None:
    st.info("Upload a SMILES CSV to start. Example columns: SMILES, Name, pIC50.")
    st.stop()

# ========== Load data ==========
try:
    raw_df = pd.read_csv(uploaded_file, header=0)
except Exception:
    # try reading without header; fallback
    uploaded_file.seek(0)
    raw_df = pd.read_csv(uploaded_file, header=None)

try:
    df = read_and_prepare(raw_df)
except Exception as e:
    st.error(f"Failed to read/prepare file: {e}")
    st.stop()

st.success(f"Loaded {len(df)} molecules.")
with st.expander("Show input table (first 20 rows)"):
    st.dataframe(df[["SMILES","Name","pIC50"]].head(20))

# ========== Run decomposition and MMP pipeline ==========
with st.spinner("Decomposing molecules into core + R-groups..."):
    row_df = decompose_all(df)
st.write(f"Fragments produced: {len(row_df)} rows.")
with st.expander("Show fragment decomposition (first 30 rows)"):
    st.dataframe(row_df.head(30))

with st.spinner("Building matched molecular pairs (MMP) delta table..."):
    delta_df = build_delta_df(row_df)
st.write(f"Pairs sharing same core: {len(delta_df)}")
if delta_df.empty:
    st.error("No pairs found that share the same core. Check your input or lower min transform occurrence.")
    st.stop()

with st.expander("Show delta table (first 50 rows)"):
    st.dataframe(delta_df.head(50))

with st.spinner("Aggregating transforms into MMP table..."):
    mmp_df, delta_df = build_mmp_df(delta_df, min_transform_occurrence=min_occ)
if mmp_df.empty:
    st.warning(f"No transforms found with min occurrence >= {min_occ}. Try lowering the threshold.")
    st.stop()

st.write(f"Found {len(mmp_df)} frequent transforms (occurrence >= {min_occ}).")
with st.expander("Show MMP DataFrame (first 50 rows)"):
    show_df = mmp_df[["Transform","Count","mean_delta"]].copy()
    show_df["mean_delta"] = show_df["mean_delta"].round(3)
    st.dataframe(show_df.head(50))

# ========== Top positive / negative transforms ==========
st.header("Top transforms — positive (increase pIC50) and negative (decrease pIC50)")

col1, col2 = st.columns(2)

with col1:
    st.subheader(f"Top {show_top_n} positive transforms")
    top_pos = mmp_df.sort_values("mean_delta", ascending=False).head(show_top_n)
    for _, row in top_pos.iterrows():
        st.markdown(f"**Transform:** `{row['Transform']}`  \n**Count:** {row['Count']}  **Mean ΔpIC50:** {row['mean_delta']:.3f}")
        img_bytes = rxn_image_bytes(row["rxn_mol"])
        if img_bytes is not None:
            st.image(img_bytes, width=300)
        # plot distribution inline
        fig, ax = plt.subplots(figsize=(4,1.2))
        plot_delta_strip(row["Deltas"], ax=ax)
        st.pyplot(fig)
        st.write("---")

with col2:
    st.subheader(f"Top {show_top_n} negative transforms")
    top_neg = mmp_df.sort_values("mean_delta", ascending=True).head(show_top_n)
    for _, row in top_neg.iterrows():
        st.markdown(f"**Transform:** `{row['Transform']}`  \n**Count:** {row['Count']}  **Mean ΔpIC50:** {row['mean_delta']:.3f}")
        img_bytes = rxn_image_bytes(row["rxn_mol"])
        if img_bytes is not None:
            st.image(img_bytes, width=300)
        fig, ax = plt.subplots(figsize=(4,1.2))
        plot_delta_strip(row["Deltas"], ax=ax)
        st.pyplot(fig)
        st.write("---")

# ========== Interactive: choose transform to inspect examples ==========
st.header("Inspect examples for a selected transform")
sel_transform = st.selectbox("Pick a transform (or search paste)", options=mmp_df.Transform.tolist())
if sel_transform:
    sel_row = mmp_df[mmp_df.Transform == sel_transform].iloc[0]
    st.markdown(f"**Selected transform:** `{sel_transform}`  \n**Count:** {sel_row['Count']}  **Mean ΔpIC50:** {sel_row['mean_delta']:.3f}")
    img_bytes = rxn_image_bytes(sel_row["rxn_mol"])
    if img_bytes is not None:
        st.image(img_bytes, width=350)

    # show all pairs (from delta_df)
    ex_pairs = delta_df[delta_df.idx == sel_row.idx].copy()
    # display a compact example table with SMILES + names + delta
    display_df = ex_pairs[["SMILES_1","Name_1","pIC50_1","SMILES_2","Name_2","pIC50_2","Delta"]].copy()
    st.dataframe(display_df.head(50))
    # downloadable CSV / excel
    csv = display_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download example pairs CSV", csv, file_name="mmp_examples.csv", mime="text/csv")

# ========== Utilities: download entire outputs ==========
st.header("Download results")

col_a, col_b = st.columns(2)
with col_a:
    csv_all_mmp = mmp_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download mmp_df (CSV)", csv_all_mmp, file_name="mmp_df.csv", mime="text/csv")
with col_b:
    csv_delta = delta_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download delta_df (CSV)", csv_delta, file_name="delta_df.csv", mime="text/csv")

# offer XLSX
def to_excel_bytes(dfs: dict):
    """dfs: name->df"""
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        for name, d in dfs.items():
            d.to_excel(writer, sheet_name=name[:31], index=False)
        writer.save()
    return out.getvalue()

xlsx_bytes = to_excel_bytes({"mmp_df": mmp_df, "delta_df": delta_df})
st.download_button("Download all results (Excel)", xlsx_bytes, file_name="mmp_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.write("---")
st.markdown("App adapted from your Colab MMP analysis notebook. If you'd like, I can add: (1) molecule thumbnails in tables, (2) more flexible fragmentation (maxCuts>1), (3) interactive plotting of transform-level statistics, or (4) saving large combined figures (png) as in the notebook.")
