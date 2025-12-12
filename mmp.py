import streamlit as st
import pandas as pd
import sys
import io
import os
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdMMPA
from operator import itemgetter
from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt
import mols2grid
import streamlit.components.v1 as components

# ---------------------------------------------------------
# 1. SETUP & CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(page_title="MMP Analysis App", layout="wide")

# ---------------------------------------------------------
# 2. HELPER FUNCTIONS
# ---------------------------------------------------------

def get_largest_fragment(mol):
    """Replicates useful_rdkit_utils.get_largest_fragment"""
    frags = Chem.GetMolFrags(mol, asMols=True)
    if not frags: return mol
    return max(frags, key=lambda x: x.GetNumAtoms())

def remove_map_nums(mol):
    for atm in mol.GetAtoms():
        atm.SetAtomMapNum(0)

def rxn_to_image(rxn_mol):
    """Helper to convert reaction to image for display"""
    return Draw.ReactionToImage(rxn_mol, subImgSize=(300, 150))

# ---------------------------------------------------------
# 3. MAIN DATA PROCESSING PIPELINE
# ---------------------------------------------------------

@st.cache_data
def process_data(df, smiles_col, name_col, act_col):
    """Step 1: Preprocessing & Fragmentation"""
    
    # Preprocessing
    df['mol'] = df[smiles_col].apply(Chem.MolFromSmiles)
    # Remove rows where mol creation failed
    df = df.dropna(subset=['mol'])
    
    # Keep only largest fragment for consistency
    df['mol'] = df['mol'].apply(get_largest_fragment)
    
    # Decompose to Scaffolds
    row_list = []
    
    # Using a progress bar for the loop
    progress_bar = st.progress(0)
    total_rows = len(df)
    
    for i, (idx, row) in enumerate(df.iterrows()):
        smiles = row[smiles_col]
        name = row[name_col]
        pIC50 = row[act_col]
        mol = row['mol']
        
        # Use RDKit's rdMMPA directly instead of scaffold_finder.py
        # FragmentMol returns a list of tuples: [(core, side_chain), ...]
        try:
            frag_list = rdMMPA.FragmentMol(mol, maxCuts=1, resultsAsMols=True)
        except Exception:
            frag_list = []
        
        for core, side_chain in frag_list:
            # The notebook logic sorts fragments by size (Core is largest)
            pair_list = [core, side_chain]
            
            # Clean map numbers for cleaner SMILES
            for frag in pair_list:
                remove_map_nums(frag)
                
            # Sort: Largest (Core) first, Smallest (R-group) second
            pair_list.sort(key=lambda x: x.GetNumAtoms(), reverse=True)
            
            # Convert to SMILES
            cores_and_r = [Chem.MolToSmiles(x) for x in pair_list]
            
            # Ensure we have exactly 2 parts (Core and R-group)
            if len(cores_and_r) == 2:
                tmp_list = [smiles, cores_and_r[0], cores_and_r[1], name, pIC50]
                row_list.append(tmp_list)
        
        if i % 10 == 0:
            progress_bar.progress(min(i / total_rows, 1.0))
            
    progress_bar.empty()
    
    row_df = pd.DataFrame(row_list, columns=["SMILES", "Core", "R_group", "Name", "pIC50"])
    return row_df

@st.cache_data
def find_pairs(row_df):
    """Step 2: Collect Pairs with Same Scaffold"""
    delta_list = []
    
    # Group by Core to find matched pairs
    grouped = row_df.groupby("Core")
    progress_bar = st.progress(0)
    total_groups = len(grouped)
    
    for i, (k, v) in enumerate(grouped):
        # We need at least 2 molecules with the same core to make a pair
        if len(v) > 1: 
            for a, b in combinations(range(0, len(v)), 2):
                reagent_a = v.iloc[a]
                reagent_b = v.iloc[b]
                
                if reagent_a.SMILES == reagent_b.SMILES:
                    continue
                
                # Canonical ordering by SMILES ensures A->B and B->A aren't duplicates
                reagent_a, reagent_b = sorted([reagent_a, reagent_b], key=lambda x: x.SMILES)
                
                delta = reagent_b.pIC50 - reagent_a.pIC50
                
                # Create the transformation string
                # Note: We replace '*' with '*-' to make it look like the notebook format if needed, 
                # but standard SMILES usually just have '*'. 
                # The notebook uses .replace('*','*-') for display.
                r1 = reagent_a.R_group.replace('*', '*-')
                r2 = reagent_b.R_group.replace('*', '*-')
                transform = f"{r1}>>{r2}"
                
                delta_list.append(
                    list(reagent_a.values) + 
                    list(reagent_b.values) + 
                    [transform, delta]
                )
        if i % 50 == 0:
             progress_bar.progress(min(i / total_groups, 1.0))

    progress_bar.empty()

    cols = ["SMILES_1", "Core_1", "R_group_1", "Name_1", "pIC50_1",
            "SMILES_2", "Core_2", "Rgroup_2", "Name_2", "pIC50_2", 
            "Transform", "Delta"]
    
    delta_df = pd.DataFrame(delta_list, columns=cols)
    return delta_df

@st.cache_data
def analyze_transforms(delta_df, min_occurrence=5):
    """Step 3: Summarize Transforms"""
    mmp_list = []
    
    # Group by the Transformation string to calculate statistics
    for k, v in delta_df.groupby("Transform"):
        if len(v) >= min_occurrence:
            mmp_list.append([k, len(v), v.Delta.values])
            
    mmp_df = pd.DataFrame(mmp_list, columns=["Transform", "Count", "Deltas"])
    
    if mmp_df.empty:
        return mmp_df
        
    # Calculate mean Delta
    mmp_df['mean_delta'] = mmp_df['Deltas'].apply(lambda x: x.mean())
    
    return mmp_df

# ---------------------------------------------------------
# 4. STREAMLIT UI LAYOUT
# ---------------------------------------------------------

st.title("MMP Analysis App")
st.markdown("""
This app replicates the Matched Molecular Pair (MMP) analysis process [web:1].
Upload your dataset to identify structural transformations influencing pIC50 values.
""")

# Sidebar Inputs
with st.sidebar:
    st.header("Settings")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file:
        # Quick peek to let user select columns
        df_preview = pd.read_csv(uploaded_file, nrows=5)
        cols = df_preview.columns.tolist()
        
        smiles_col = st.selectbox("SMILES Column", cols, index=cols.index("SMILES") if "SMILES" in cols else 0)
        name_col = st.selectbox("Name/ID Column", cols, index=cols.index("Name") if "Name" in cols else 0)
        act_col = st.selectbox("Activity (pIC50) Column", cols, index=cols.index("pIC50") if "pIC50" in cols else 0)
        
        min_occurrence = st.number_input("Min Transform Occurrence", value=5, min_value=1)
        run_btn = st.button("Run Analysis")

# Main Execution
if uploaded_file and run_btn:
    # Load Data
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file)
    
    st.info(f"Loaded {len(df)} molecules. Starting Fragmentation...")
    
    # 1. Process Fragments
    row_df = process_data(df, smiles_col, name_col, act_col)
    if row_df.empty:
        st.error("No fragments generated. Please check your SMILES column.")
    else:
        st.write(f"Generated {len(row_df)} fragments.")
        
        # 2. Find Pairs
        st.info("Finding Matched Pairs...")
        delta_df = find_pairs(row_df)
        st.write(f"Found {len(delta_df)} matched pairs.")
        
        if delta_df.empty:
             st.warning("No matched pairs found. Try a larger dataset or check for similar scaffolds.")
        else:
            # 3. Analyze
            st.info("Analyzing Transformations...")
            mmp_df = analyze_transforms(delta_df, min_occurrence)
            
            if mmp_df.empty:
                st.warning(f"No transformations found with at least {min_occurrence} occurrences.")
            else:
                st.success(f"Identified {len(mmp_df)} unique transformations.")

                # ---------------------------------------------------------
                # RESULTS DISPLAY
                # ---------------------------------------------------------
                
                tab1, tab2, tab3 = st.tabs(["Top Transformations", "All Transforms Table", "Downloads"])

                with tab1:
                    st.subheader("Top Positive Transformations (Potency Enhancers)")
                    top_pos = mmp_df.sort_values("mean_delta", ascending=False).head(3)
                    
                    for i, (idx, row) in enumerate(top_pos.iterrows()):
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            st.markdown(f"**Rank {i+1}:** `{row['Transform']}`")
                            st.write(f"**Mean ΔpIC50:** {row['mean_delta']:.2f}")
                            st.write(f"**Count:** {row['Count']}")
                            
                            # Generate Reaction Image
                            try:
                                rxn = AllChem.ReactionFromSmarts(row['Transform'].replace('*-','*'), useSmiles=True)
                                st.image(rxn_to_image(rxn), use_container_width=True)
                            except:
                                st.warning("Could not generate image for this transform")
                            
                        with col2:
                            # Strip Plot
                            fig, ax = plt.subplots(figsize=(4, 2))
                            sns.stripplot(x=row['Deltas'], ax=ax, color='blue', alpha=0.6, jitter=0.2)
                            ax.axvline(0, ls="--", c="red")
                            ax.set_xlim(-5, 5)
                            ax.set_title("ΔpIC50 Distribution")
                            st.pyplot(fig)
                            plt.close(fig)

                        # Show examples for this transform
                        with st.expander(f"Show Compound Pairs for Rank {i+1}"):
                            examples = delta_df[delta_df['Transform'] == row['Transform']].head(6)
                            
                            grid_data = []
                            for _, ex_row in examples.iterrows():
                                grid_data.append({
                                    "SMILES": ex_row['SMILES_1'], 
                                    "Name": f"{ex_row['Name_1']} (A)", 
                                    "pIC50": ex_row['pIC50_1'],
                                })
                                grid_data.append({
                                    "SMILES": ex_row['SMILES_2'], 
                                    "Name": f"{ex_row['Name_2']} (B)", 
                                    "pIC50": ex_row['pIC50_2'],
                                })
                            
                            grid_df = pd.DataFrame(grid_data)
                            raw_html = mols2grid.display(
                                grid_df, 
                                smiles_col="SMILES", 
                                subset=["img", "Name", "pIC50"],
                                n_cols=4, 
                                size=(150, 150)
                            )._repr_html_()
                            components.html(raw_html, height=400, scrolling=True)

                    st.markdown("---")
                    st.subheader("Top Negative Transformations (Potency Diminishers)")
                    top_neg = mmp_df.sort_values("mean_delta", ascending=True).head(3)
                    
                    for i, (idx, row) in enumerate(top_neg.iterrows()):
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            st.markdown(f"**Rank {i+1}:** `{row['Transform']}`")
                            st.write(f"**Mean ΔpIC50:** {row['mean_delta']:.2f}")
                            st.write(f"**Count:** {row['Count']}")
                            try:
                                rxn = AllChem.ReactionFromSmarts(row['Transform'].replace('*-','*'), useSmiles=True)
                                st.image(rxn_to_image(rxn), use_container_width=True)
                            except:
                                st.warning("Could not generate image")
                        with col2:
                            fig, ax = plt.subplots(figsize=(4, 2))
                            sns.stripplot(x=row['Deltas'], ax=ax, color='red', alpha=0.6, jitter=0.2)
                            ax.axvline(0, ls="--", c="red")
                            ax.set_xlim(-5, 5)
                            st.pyplot(fig)
                            plt.close(fig)

                with tab2:
                    st.subheader("Full Transformation Table")
                    st.dataframe(mmp_df.drop(columns=['Deltas'], errors='ignore').
