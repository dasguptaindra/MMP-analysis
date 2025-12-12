import streamlit as st
import pandas as pd
import io
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdMMPA
from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt
import mols2grid
import streamlit.components.v1 as components

# ---------------------------------------------------------
# 1. HELPER FUNCTIONS
# ---------------------------------------------------------

def get_largest_fragment(mol):
    """Return largest fragment to handle salts/solvents."""
    if mol is None: return None
    frags = Chem.GetMolFrags(mol, asMols=True)
    if not frags: return mol
    return max(frags, key=lambda x: x.GetNumAtoms())

def remove_map_nums(mol):
    """Remove atom mapping numbers for clean SMILES generation."""
    if mol is None: return
    for atm in mol.GetAtoms():
        atm.SetAtomMapNum(0)

def rxn_to_image(rxn_mol):
    """Convert RDKit reaction object to image."""
    return Draw.ReactionToImage(rxn_mol, subImgSize=(300, 150))

# ---------------------------------------------------------
# 2. CORE LOGIC
# ---------------------------------------------------------

@st.cache_data
def process_data(df, smiles_col, name_col, act_col):
    """Preprocessing and Fragmentation using rdMMPA."""
    # 1. Clean and convert to Mol
    df = df.dropna(subset=[smiles_col])
    df['mol'] = df[smiles_col].astype(str).apply(Chem.MolFromSmiles)
    df = df.dropna(subset=['mol'])
    df['mol'] = df['mol'].apply(get_largest_fragment)
    
    row_list = []
    progress_bar = st.progress(0)
    total_rows = len(df)
    
    for i, (idx, row) in enumerate(df.iterrows()):
        mol = row['mol']
        # Try to fragment
        try:
            # resultsAsMols=True ensures we get objects, not strings
            frags = rdMMPA.FragmentMol(mol, maxCuts=1, resultsAsMols=True)
        except:
            frags = []
        
        for frag_pair in frags:
            # We expect a tuple of (Core, Sidechain)
            if len(frag_pair) != 2: continue
            
            # Convert to Mol objects if they aren't already
            mols_pair = []
            for part in frag_pair:
                if isinstance(part, str):
                    m = Chem.MolFromSmiles(part)
                    if not m: m = Chem.MolFromSmarts(part)
                    mols_pair.append(m)
                else:
                    mols_pair.append(part)
            
            if len(mols_pair) != 2 or any(x is None for x in mols_pair): continue
            
            # Clean and Sort: Largest (Core) first
            for m in mols_pair: remove_map_nums(m)
            mols_pair.sort(key=lambda x: x.GetNumAtoms(), reverse=True)
            
            try:
                cores_and_r = [Chem.MolToSmiles(x) for x in mols_pair]
                if len(cores_and_r) == 2:
                    # [SMILES, Core, R_group, Name, pIC50]
                    row_list.append([row[smiles_col], cores_and_r[0], cores_and_r[1], row[name_col], row[act_col]])
            except: continue

        if i % 10 == 0: progress_bar.progress(min(i / total_rows, 1.0))
            
    progress_bar.empty()
    return pd.DataFrame(row_list, columns=["SMILES", "Core", "R_group", "Name", "pIC50"])

@st.cache_data
def find_pairs(row_df):
    """Find pairs of molecules sharing the same core."""
    delta_list = []
    grouped = row_df.groupby("Core")
    progress_bar = st.progress(0)
    total_groups = len(grouped)
    
    for i, (k, v) in enumerate(grouped):
        if len(v) > 1:
            for a, b in combinations(range(len(v)), 2):
                ra, rb = v.iloc[a], v.iloc[b]
                if ra.SMILES == rb.SMILES: continue
                
                # Sort by SMILES for canonical order
                ra, rb = sorted([ra, rb], key=lambda x: x.SMILES)
                delta = rb.pIC50 - ra.pIC50
                
                # Transform string: R1>>R2
                r1 = ra.R_group.replace('*', '*-')
                r2 = rb.R_group.replace('*', '*-')
                
                delta_list.append(list(ra.values) + list(rb.values) + [f"{r1}>>{r2}", delta])
        
        if i % 50 == 0: progress_bar.progress(min(i / total_groups, 1.0))

    progress_bar.empty()
    cols = ["SMILES_1", "Core_1", "R_group_1", "Name_1", "pIC50_1",
            "SMILES_2", "Core_2", "Rgroup_2", "Name_2", "pIC50_2", "Transform", "Delta"]
    return pd.DataFrame(delta_list, columns=cols)

@st.cache_data
def analyze_transforms(delta_df, min_occurrence):
    """Aggregate stats for each transformation."""
    mmp_list = []
    for k, v in delta_df.groupby("Transform"):
        if len(v) >= min_occurrence:
            mmp_list.append([k, len(v), v.Delta.values, v.Delta.mean()])
    return pd.DataFrame(mmp_list, columns=["Transform", "Count", "Deltas", "mean_delta"])

# ---------------------------------------------------------
# 3. UI LAYOUT
# ---------------------------------------------------------

st.set_page_config(page_title="MMP Analysis App", layout="wide")
st.title("MMP Analysis App")

with st.sidebar:
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df_preview = pd.read_csv(uploaded_file, nrows=5)
        cols = df_preview.columns.tolist()
        smiles_col = st.selectbox("SMILES Column", cols, index=cols.index("SMILES") if "SMILES" in cols else 0)
        name_col = st.selectbox("Name/ID Column", cols, index=cols.index("Name") if "Name" in cols else 0)
        act_col = st.selectbox("pIC50 Column", cols, index=cols.index("pIC50") if "pIC50" in cols else 0)
        min_occ = st.number_input("Min Occurrences", value=5, min_value=1)
        run_btn = st.button("Run Analysis")

if uploaded_file and 'run_btn' in locals() and run_btn:
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file)
    
    st.info("Fragmenting molecules...")
    row_df = process_data(df, smiles_col, name_col, act_col)
    
    if row_df.empty:
        st.error("No valid fragments generated.")
    else:
        st.write(f"Generated {len(row_df)} fragments. Finding pairs...")
        delta_df = find_pairs(row_df)
        
        if delta_df.empty:
            st.warning("No matched pairs found.")
        else:
            st.write(f"Found {len(delta_df)} pairs. Analyzing...")
            mmp_df = analyze_transforms(delta_df, min_occ)
            
            if mmp_df.empty:
                st.warning("No transformations met the occurrence threshold.")
            else:
                st.success(f"Found {len(mmp_df)} unique transformations.")
                
                # Tabs
                t1, t2, t3 = st.tabs(["Top Transformations", "Full Table", "Export"])
                
                with t1:
                    st.subheader("Top Positive (Potency Increasing)")
                    top_pos = mmp_df.sort_values("mean_delta", ascending=False).head(3)
                    for _, row in top_pos.iterrows():
                        c1, c2 = st.columns([1, 1])
                        c1.markdown(f"**{row['Transform']}**\n\nMean Δ: {row['mean_delta']:.2f}, Count: {row['Count']}")
                        try: c1.image(rxn_to_image(AllChem.ReactionFromSmarts(row['Transform'].replace('*-','*'), useSmiles=True)))
                        except: pass
                        
                        fig, ax = plt.subplots(figsize=(4, 1.5))
                        sns.stripplot(x=row['Deltas'], ax=ax, color='blue', alpha=0.6)
                        ax.axvline(0, ls="--", c="red")
                        c2.pyplot(fig)
                        plt.close(fig)

                    st.markdown("---")
                    st.subheader("Top Negative (Potency Decreasing)")
                    top_neg = mmp_df.sort_values("mean_delta", ascending=True).head(3)
                    for _, row in top_neg.iterrows():
                        c1, c2 = st.columns([1, 1])
                        c1.markdown(f"**{row['Transform']}**\n\nMean Δ: {row['mean_delta']:.2f}, Count: {row['Count']}")
                        try: c1.image(rxn_to_image(AllChem.ReactionFromSmarts(row['Transform'].replace('*-','*'), useSmiles=True)))
                        except: pass
                        
                        fig, ax = plt.subplots(figsize=(4, 1.5))
                        sns.stripplot(x=row['Deltas'], ax=ax, color='red', alpha=0.6)
                        ax.axvline(0, ls="--", c="red")
                        c2.pyplot(fig)
                        plt.close(fig)

                with t2:
                    st.dataframe(mmp_df.drop(columns=['Deltas']).sort_values("mean_delta", ascending=False))
                
                with t3:
                    out = io.BytesIO()
                    mmp_export = mmp_df.copy()
                    mmp_export['Deltas'] = mmp_export['Deltas'].astype(str)
                    with pd.ExcelWriter(out, engine='openpyxl') as writer:
                        mmp_export.to_excel(writer, sheet_name='Transforms', index=False)
                        delta_df.to_excel(writer, sheet_name='Pairs', index=False)
                    st.download_button("Download Excel Report", out.getvalue(), "MMP_Report.xlsx")
