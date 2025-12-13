import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdMMPA import FragmentMol
from rdkit.Chem import AllChem
from rdkit.Chem.SaltRemover import SaltRemover
from operator import itemgetter
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from rdkit.Chem.Draw import rdMolDraw2D

# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(
    page_title="MMP Analysis Tool",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Matched Molecular Pair (MMP) Analysis")

# -----------------------------
# Helper Functions
# -----------------------------

def get_largest_fragment(mol):
    """
    Standardizes the molecule by removing salts and keeping the largest organic fragment.
    """
    if mol is None:
        return None
    try:
        remover = SaltRemover()
        mol = remover.StripMol(mol, dontRemoveEverything=True)
        frags = list(Chem.GetMolFrags(mol, asMols=True))
        if not frags:
            return mol
        # Return the fragment with the most atoms
        return max(frags, key=lambda m: m.GetNumAtoms())
    except Exception:
        return mol

def remove_map_nums(mol):
    for atm in mol.GetAtoms():
        atm.SetAtomMapNum(0)

def sort_fragments(mol):
    """
    Splits the fragmented molecule into constituent parts 
    and returns them sorted by size (largest first -> Core).
    """
    if mol is None:
        return []
    try:
        frag_list = list(Chem.GetMolFrags(mol, asMols=True))
        for x in frag_list:
            remove_map_nums(x)
        # Sort by number of atoms (descending)
        frag_num_atoms_list = [(x.GetNumAtoms(), x) for x in frag_list]
        frag_num_atoms_list.sort(key=itemgetter(0), reverse=True)
        return [x[1] for x in frag_num_atoms_list]
    except Exception:
        return []

def rxn_to_base64_image(rxn):
    """Generates a base64 encoded SVG of the reaction."""
    try:
        drawer = rdMolDraw2D.MolDraw2DSVG(300, 150)
        drawer.DrawReaction(rxn)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        b64 = base64.b64encode(svg.encode()).decode()
        return f'<img src="data:image/svg+xml;base64,{b64}"/>'
    except:
        return ""

def stripplot_base64_image(deltas):
    """Generates a base64 encoded PNG of the stripplot."""
    try:
        fig, ax = plt.subplots(figsize=(3, 2))
        sns.stripplot(y=deltas, ax=ax, color="#4e79a7", alpha=0.7)
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax.set_ylabel("Delta pIC50")
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        return f'<img src="data:image/png;base64,{b64}"/>'
    except:
        return ""

# -----------------------------
# Sidebar & Input
# -----------------------------
with st.sidebar:
    st.header("Settings")
    min_transform_occurrence = st.slider(
        "Minimum MMP Occurrence", 2, 50, 3, 
        help="Only show transformations that appear at least this many times in the dataset."
    )
    
    uploaded_file = st.file_uploader(
        "Upload CSV (must contain SMILES, Name, pIC50)", type=["csv"]
    )

# -----------------------------
# Main Logic
# -----------------------------
if uploaded_file:
    # 1. Load Data
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()
    
    # 2. Column Mapping (Robustness)
    st.subheader("1. Data Mapping")
    cols = df.columns.tolist()
    
    c1, c2, c3 = st.columns(3)
    smiles_col = c1.selectbox("Select SMILES column", cols, index=cols.index("SMILES") if "SMILES" in cols else 0)
    name_col = c2.selectbox("Select ID/Name column", cols, index=cols.index("Name") if "Name" in cols else 0)
    pic50_col = c3.selectbox("Select Activity (pIC50) column", cols, index=cols.index("pIC50") if "pIC50" in cols else 0)

    # Validate numeric activity
    df = df.dropna(subset=[smiles_col, pic50_col])
    df[pic50_col] = pd.to_numeric(df[pic50_col], errors='coerce')
    df = df.dropna(subset=[pic50_col])

    st.info(f"Loaded {len(df)} valid molecules.")

    # 3. Processing
    if st.button("Run MMP Analysis"):
        with st.spinner("Fragmenting molecules..."):
            # Prepare Mols
            df["mol"] = df[smiles_col].apply(Chem.MolFromSmiles)
            df = df.dropna(subset=["mol"])
            # Apply standardization (replacing useful_rdkit_utils)
            df["mol"] = df["mol"].apply(get_largest_fragment)

            # Fragmentation Loop
            row_list = []
            
            for idx, row in df.iterrows():
                mol = row["mol"]
                smiles = row[smiles_col]
                name = row[name_col]
                pIC50 = row[pic50_col]
                
                if mol:
                    # Robust FragmentMol call
                    try:
                        # resultsAsMols=True returns the fragmented molecule directly (with dummy atoms)
                        # or a sequence of them depending on RDKit version.
                        frags = FragmentMol(mol, maxCuts=1, resultsAsMols=True)
                    except Exception:
                        continue

                    # Iterate safely over whatever FragmentMol returns
                    for frag_item in frags:
                        # Sometimes it returns (num_cuts, mol), sometimes just mol. Check type:
                        if isinstance(frag_item, tuple):
                            frag_mol = frag_item[1] # Use the molecule part
                        else:
                            frag_mol = frag_item
                        
                        # Now process the fragmented molecule
                        pair = sort_fragments(frag_mol)
                        
                        # We only want pairs (Core + R-Group)
                        if len(pair) == 2:
                            core_smi = Chem.MolToSmiles(pair[0]) # Largest is core
                            r_smi = Chem.MolToSmiles(pair[1])    # Smallest is R
                            
                            row_list.append([
                                smiles, core_smi, r_smi, name, pIC50
                            ])

            row_df = pd.DataFrame(
                row_list,
                columns=["SMILES", "Core", "R_group", "Name", "pIC50"]
            )

        if row_df.empty:
            st.error("No valid fragments generated. Please check if your SMILES are valid or increase the dataset size.")
        else:
            # 4. Delta Calculation
            with st.spinner("Calculating Deltas..."):
                delta_rows = []
                # Group by Core to find pairs
                grouped = row_df.groupby("Core")
                
                for core, group in grouped:
                    if len(group) >= 2:
                        # Iterate all combinations in this core group
                        # Combinations of INDICES to avoid re-sorting large dataframes
                        group_indices = range(len(group))
                        for i, j in combinations(group_indices, 2):
                            ra = group.iloc[i]
                            rb = group.iloc[j]
                            
                            # Avoid self-matches and identical R-groups
                            if ra.R_group == rb.R_group:
                                continue
                            
                            # Canonical ordering by SMILES to ensure A>>B is same as B>>A in direction logic
                            mols_sorted = sorted([ra, rb], key=lambda x: x.R_group)
                            r1, r2 = mols_sorted[0], mols_sorted[1]
                            
                            transform_str = f"{r1.R_group.replace('*','*-')}>>{r2.R_group.replace('*','*-')}"
                            delta_val = r2.pIC50 - r1.pIC50
                            
                            delta_rows.append([
                                r1.SMILES, r1.Core, r1.R_group, r1.Name, r1.pIC50,
                                r2.SMILES, r2.Core, r2.R_group, r2.Name, r2.pIC50,
                                transform_str, delta_val
                            ])

                delta_df = pd.DataFrame(delta_rows, columns=[
                    "SMILES_1", "Core_1", "R_group_1", "Name_1", "pIC50_1",
                    "SMILES_2", "Core_2", "R_group_2", "Name_2", "pIC50_2",
                    "Transform", "Delta"
                ])

            # 5. Aggregation & Display
            st.subheader("Final MMP Results")
            
            if delta_df.empty:
                st.warning("No matched pairs found.")
            else:
                mmp_rows = []
                for transform, v in delta_df.groupby("Transform"):
                    if len(v) >= min_transform_occurrence:
                        mmp_rows.append([transform, len(v), v.Delta.values])

                if not mmp_rows:
                    st.warning(f"No transformations found with occurrence >= {min_transform_occurrence}.")
                else:
                    mmp_df = pd.DataFrame(mmp_rows, columns=["Transform", "Count", "Deltas"])
                    mmp_df["mean_delta"] = mmp_df["Deltas"].apply(np.mean)
                    
                    # Create Reaction Objects from Transform string
                    # Note: Using simple string replacement to valid SMARTS/SMILES for reaction
                    mmp_df["rxn"] = mmp_df["Transform"].apply(
                        lambda x: AllChem.ReactionFromSmarts(x.replace("*-", "*"), useSmiles=True)
                    )

                    # Build HTML Visuals
                    mmp_df["MMP Transform"] = mmp_df["rxn"].apply(rxn_to_base64_image)
                    mmp_df["Delta Distribution"] = mmp_df["Deltas"].apply(stripplot_base64_image)
                    
                    # Formatting for Display
                    display_cols = ["MMP Transform", "Count", "mean_delta", "Delta Distribution"]
                    
                    # Sort by count
                    mmp_df = mmp_df.sort_values("Count", ascending=False)

                    # Generate HTML Table
                    html_table = (
                        mmp_df[display_cols]
                        .round(2)
                        .to_html(escape=False, index=False)
                    )
                    
                    # Custom CSS to center images in table
                    st.markdown("""
                    <style>
                    table td { vertical-align: middle !important; text-align: center !important; }
                    table th { text-align: center !important; }
                    </style>
                    """, unsafe_allow_html=True)

                    components.html(html_table, height=800, scrolling=True)

else:
    st.info("Upload a CSV file to start analysis.")
