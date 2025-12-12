import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMMPA, AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from tqdm import tqdm
from itertools import combinations
import base64
import io
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------------------
# Helper Functions (Replicating logic from the notebook)
# ------------------------------------------------------------------------------

def get_largest_fragment(mol):
    """
    Returns the largest fragment of a molecule (filters out salts/solvents).
    """
    frags = list(Chem.GetMolFrags(mol, asMols=True))
    frags.sort(key=lambda x: x.GetNumAtoms(), reverse=True)
    return frags[0]

def remove_map_nums(mol):
    """
    Remove atom map numbers from a molecule.
    """
    for atm in mol.GetAtoms():
        atm.SetAtomMapNum(0)
    return mol

def mol_to_base64(mol, size=(200, 200)):
    """
    Convert RDKit molecule to base64 PNG string for HTML display.
    """
    if mol is None:
        return ""
    img = Draw.MolToImage(mol, size=size)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def rxn_to_base64(rxn_smarts, size=(400, 150)):
    """
    Convert a Reaction SMARTS to a base64 image.
    """
    try:
        rxn = AllChem.ReactionFromSmarts(rxn_smarts, useSmiles=True)
        d2d = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        d2d.DrawReaction(rxn)
        d2d.FinishDrawing()
        return base64.b64encode(d2d.GetDrawingText()).decode("utf-8")
    except:
        return ""

def process_mmp(df, pIC50_col='pIC50', smiles_col='SMILES', name_col='Name'):
    """
    Performs the MMP fragmentation and pairing analysis.
    """
    # 1. Clean and Standardize Molecules
    mols = []
    valid_indices = []
    
    st.write("Preprocessing molecules...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        smi = row[smiles_col]
        mol = Chem.MolFromSmiles(smi)
        if mol:
            # Keep largest fragment (remove salts)
            mol = get_largest_fragment(mol)
            mols.append(mol)
            valid_indices.append(idx)
    
    df_clean = df.iloc[valid_indices].copy()
    df_clean['mol'] = mols
    
    # 2. Fragment Molecules
    st.write("Generating fragments...")
    row_list = []
    
    # Iterate over molecules
    for i, row in tqdm(df_clean.iterrows(), total=len(df_clean)):
        mol = row['mol']
        name = row[name_col]
        pic50 = row[pIC50_col]
        smiles = row[smiles_col]
        
        # MMPA Fragmentation (Single cut)
        # rdMMPA returns list of tuples: (core, side_chain)
        frags = rdMMPA.FragmentMol(mol, maxCuts=1, resultsAsMols=True)
        
        for core, chain in frags:
            # The notebook sorts fragments by size to define Core vs R-group
            # We treat the larger piece as Core
            parts = [core, chain]
            # Clean map numbers (dummies) for consistent SMILES
            # Note: keeping dummies is essential for the reaction SMARTS later, 
            # but the notebook removes map numbers for the 'Core' column to group by.
            
            # Create copies for processing
            p1 = remove_map_nums(Chem.Mol(parts[0]))
            p2 = remove_map_nums(Chem.Mol(parts[1]))
            
            # Sort by atom count to identify Core (largest)
            if p1.GetNumAtoms() >= p2.GetNumAtoms():
                core_smi = Chem.MolToSmiles(p1)
                r_smi = Chem.MolToSmiles(p2)
            else:
                core_smi = Chem.MolToSmiles(p2)
                r_smi = Chem.MolToSmiles(p1)
                
            row_list.append([smiles, core_smi, r_smi, name, pic50])

    row_df = pd.DataFrame(row_list, columns=["SMILES", "Core", "R_group", "Name", "pIC50"])
    
    # 3. Find Pairs (Same Core, Different R-group)
    st.write("Identifying matched pairs...")
    delta_list = []
    
    # Group by Core
    groups = row_df.groupby("Core")
    
    for core, group in tqdm(groups):
        if len(group) >= 2:
            # Generate all pairs in this group
            for idx_a, idx_b in combinations(group.index, 2):
                row_a = group.loc[idx_a]
                row_b = group.loc[idx_b]
                
                # Skip identical compounds
                if row_a.SMILES == row_b.SMILES:
                    continue
                
                # Sort pair to ensure consistent direction (or just A->B)
                # The notebook sorts by SMILES to make order canonical
                if row_a.SMILES > row_b.SMILES:
                    row_a, row_b = row_b, row_a
                
                delta = row_b.pIC50 - row_a.pIC50
                
                # Define Transform: R_group_A >> R_group_B
                transform = f"{row_a.R_group}>>{row_b.R_group}"
                
                delta_list.append({
                    "SMILES_1": row_a.SMILES, "Name_1": row_a.Name, "pIC50_1": row_a.pIC50, "R_group_1": row_a.R_group,
                    "SMILES_2": row_b.SMILES, "Name_2": row_b.Name, "pIC50_2": row_b.pIC50, "R_group_2": row_b.R_group,
                    "Core": core,
                    "Transform": transform,
                    "Delta": delta
                })
                
    delta_df = pd.DataFrame(delta_list)
    return delta_df

# ------------------------------------------------------------------------------
# Main App Layout
# ------------------------------------------------------------------------------

st.set_page_config(layout="wide", page_title="MMP Analysis")

st.title("Matched Molecular Pair (MMP) Analysis")
st.markdown("""
This app replicates the MMP analysis workflow. 
Upload a CSV containing **SMILES**, **Name**, and **pIC50** columns.
""")

# File Uploader
uploaded_file = st.file_uploader("Upload Molecule CSV", type=["csv"])

if uploaded_file:
    # Load Data
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head(), use_container_width=True)
    
    # Column Mapping
    cols = df.columns.tolist()
    c1, c2, c3 = st.columns(3)
    smiles_col = c1.selectbox("SMILES Column", cols, index=cols.index("SMILES") if "SMILES" in cols else 0)
    name_col = c2.selectbox("Name/ID Column", cols, index=cols.index("Name") if "Name" in cols else 0)
    pic50_col = c3.selectbox("pIC50/Activity Column", cols, index=cols.index("pIC50") if "pIC50" in cols else 0)
    
    if st.button("Run Analysis"):
        with st.spinner("Running MMP Analysis..."):
            # Run Logic
            delta_df = process_mmp(df, pic50_col, smiles_col, name_col)
            
            if delta_df.empty:
                st.warning("No matched pairs found.")
            else:
                st.success(f"Found {len(delta_df)} matched pairs.")
                
                # ----------------------------------------------------------------------
                # Analysis: Group by Transformation
                # ----------------------------------------------------------------------
                min_occ = st.slider("Minimum Occurrences for Transformation", 2, 20, 5)
                
                # Aggregate stats
                mmp_stats = []
                for transform, group in delta_df.groupby("Transform"):
                    if len(group) >= min_occ:
                        mmp_stats.append({
                            "Transform": transform,
                            "Count": len(group),
                            "Mean_Delta": group.Delta.mean(),
                            "Std_Delta": group.Delta.std(),
                            "Deltas": group.Delta.values
                        })
                
                mmp_df = pd.DataFrame(mmp_stats)
                
                if mmp_df.empty:
                    st.warning(f"No transformations found with >= {min_occ} occurrences.")
                else:
                    # Sort by Mean Delta (Desc)
                    mmp_df = mmp_df.sort_values("Mean_Delta", ascending=False).reset_index(drop=True)
                    
                    # Display Top Transformations
                    st.subheader("Top Transformations")
                    
                    # Add visuals to dataframe for display
                    display_data = []
                    for i, row in mmp_df.iterrows():
                        # Create reaction image
                        img_b64 = rxn_to_base64(row['Transform'])
                        img_html = f'<img src="data:image/png;base64,{img_b64}" width="300" />'
                        
                        display_data.append({
                            "Rank": i + 1,
                            "Transform": row['Transform'],
                            "Reaction": img_html,
                            "Count": row['Count'],
                            "Mean Delta pIC50": f"{row['Mean_Delta']:.2f}",
                            "Std Dev": f"{row['Std_Delta']:.2f}"
                        })
                    
                    # Render HTML Table for images
                    st.markdown(
                        pd.DataFrame(display_data).to_html(escape=False, index=False), 
                        unsafe_allow_html=True
                    )
                    
                    # ------------------------------------------------------------------
                    # Detailed View
                    # ------------------------------------------------------------------
                    st.markdown("---")
                    st.subheader("Transformation Detail View")
                    
                    selected_transform = st.selectbox(
                        "Select a Transformation to inspect:", 
                        mmp_df['Transform'].tolist()
                    )
                    
                    if selected_transform:
                        row = mmp_df[mmp_df['Transform'] == selected_transform].iloc[0]
                        subset = delta_df[delta_df['Transform'] == selected_transform]
                        
                        c_left, c_right = st.columns([1, 1])
                        
                        with c_left:
                            st.markdown(f"**Transform:** {selected_transform}")
                            st.markdown(f"**Count:** {row['Count']}")
                            st.markdown(f"**Mean Delta:** {row['Mean_Delta']:.2f}")
                            
                            # Reaction Image
                            st.image(io.BytesIO(base64.b64decode(rxn_to_base64(selected_transform))), caption="Transformation Reaction")
                        
                        with c_right:
                            # Strip Plot
                            fig, ax = plt.subplots(figsize=(4, 3))
                            sns.stripplot(x=row['Deltas'], ax=ax, jitter=True, color='blue', alpha=0.6)
                            ax.axvline(0, ls="--", c="red")
                            ax.set_title("Distribution of ΔpIC50")
                            ax.set_xlabel("Delta pIC50 (B - A)")
                            st.pyplot(fig)
                        
                        st.subheader("Example Pairs")
                        # Show table of examples for this transform
                        example_cols = ["Name_1", "SMILES_1", "pIC50_1", "Name_2", "SMILES_2", "pIC50_2", "Delta"]
                        st.dataframe(subset[example_cols].head(10))

