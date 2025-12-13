# -*- coding: utf-8 -*-
import streamlit as st
import sys
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from scaffold_finder import FragmentMol
from operator import itemgetter
import useful_rdkit_utils as uru
from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image
import time

# Set page configuration
st.set_page_config(
    page_title="Matched Molecular Pairs Analysis",
    page_icon="🧪",
    layout="wide"
)

# Title and description
st.title("🧪 Matched Molecular Pairs Analysis")
st.markdown("""
This application performs Matched Molecular Pairs (MMP) analysis on a hERG inhibition dataset from ChEMBL.
MMP analysis helps identify the impact of specific chemical changes on biological activity.

**References:**
1. Hussain, Jameed, and Ceara Rea. "Computationally efficient algorithm to identify matched molecular pairs (MMPs) in large data sets." *Journal of Chemical Information and Modeling*, **50** (2010): 339-348.
2. Dossetter, Alexander G., et al. "Matched molecular pair analysis in drug discovery." *Drug Discovery Today*, **18** (2013): 724-731.
""")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .stDataFrame {
        font-size: 0.9rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# Utility functions
def remove_map_nums(mol):
    """Remove atom map numbers from a molecule"""
    for atm in mol.GetAtoms():
        atm.SetAtomMapNum(0)

def sort_fragments(mol):
    """
    Transform a molecule with multiple fragments into a list of molecules 
    that is sorted by number of atoms from largest to smallest
    """
    frag_list = list(Chem.GetMolFrags(mol, asMols=True))
    [remove_map_nums(x) for x in frag_list]
    frag_num_atoms_list = [(x.GetNumAtoms(), x) for x in frag_list]
    frag_num_atoms_list.sort(key=itemgetter(0), reverse=True)
    return [x[1] for x in frag_num_atoms_list]

def rxn_to_pil_image(rxn):
    """Convert an RDKit reaction to a PIL Image"""
    drawer = rdMolDraw2D.MolDraw2DCairo(400, 200)
    drawer.DrawReaction(rxn)
    drawer.FinishDrawing()
    img_bytes = drawer.GetDrawingText()
    img = Image.open(io.BytesIO(img_bytes))
    return img

def plot_delta_distribution(dist, ax):
    """Plot a distribution as a seaborn stripplot"""
    sns.stripplot(x=dist, ax=ax, size=8, alpha=0.7)
    ax.axvline(0, ls="--", c="red", alpha=0.7)
    ax.set_xlim(-5, 5)
    ax.set_xlabel("ΔpIC50", fontsize=10)
    ax.set_title("Distribution of Activity Changes", fontsize=12)
    ax.grid(True, alpha=0.3)
    return ax

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")

# Parameters
min_transform_occurrence = st.sidebar.slider(
    "Minimum Transform Occurrence",
    min_value=1,
    max_value=20,
    value=5,
    help="Minimum number of times a transform must appear to be included in analysis"
)

rows_to_show = st.sidebar.slider(
    "Number of MMPs to Display",
    min_value=5,
    max_value=50,
    value=10,
    help="Number of top MMPs to show in results table"
)

show_ascending = st.sidebar.checkbox(
    "Show Activity-Decreasing Transforms First",
    value=True,
    help="When checked, shows transforms that decrease activity first"
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Note:** The analysis uses a pre-loaded hERG inhibition dataset from ChEMBL containing SMILES, ChEMBL IDs, and pIC50 values.
""")

# Main application
if st.button("🚀 Start Analysis") or 'analysis_done' in st.session_state:
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = True
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 1: Read and Process Input Data
    status_text.text("📥 Step 1/6: Loading data from ChEMBL...")
    try:
        df = pd.read_csv("https://raw.githubusercontent.com/PatWalters/practical_cheminformatics_tutorials/main/data/hERG.csv")
        progress_bar.progress(10)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
    
    # Step 2: Add RDKit molecules
    status_text.text("⚗️ Step 2/6: Processing molecules...")
    df['mol'] = df.SMILES.apply(Chem.MolFromSmiles)
    progress_bar.progress(20)
    
    # Step 3: Remove salts and counterions
    df.mol = df.mol.apply(uru.get_largest_fragment)
    progress_bar.progress(30)
    
    # Step 4: Decompose molecules to scaffolds and sidechains
    status_text.text("🔬 Step 3/6: Decomposing molecules into scaffolds and sidechains...")
    row_list = []
    
    # Create a progress bar for decomposition
    decomposition_progress = st.progress(0)
    total_rows = len(df)
    
    for i, (smiles, name, pIC50, mol) in enumerate(df.values):
        frag_list = FragmentMol(mol, maxCuts=1)
        for _, frag_mol in frag_list:
            pair_list = sort_fragments(frag_mol)
            tmp_list = [smiles] + [Chem.MolToSmiles(x) for x in pair_list] + [name, pIC50]
            row_list.append(tmp_list)
        decomposition_progress.progress((i + 1) / total_rows)
    
    decomposition_progress.empty()
    row_df = pd.DataFrame(row_list, columns=["SMILES", "Core", "R_group", "Name", "pIC50"])
    progress_bar.progress(50)
    
    # Step 5: Collect pairs with same scaffold
    status_text.text("🔗 Step 4/6: Finding matched molecular pairs...")
    delta_list = []
    
    # Group by core and find pairs
    groups = list(row_df.groupby("Core"))
    pair_progress = st.progress(0)
    
    for idx, (k, v) in enumerate(groups):
        if len(v) > 2:
            for a, b in combinations(range(0, len(v)), 2):
                reagent_a = v.iloc[a]
                reagent_b = v.iloc[b]
                if reagent_a.SMILES == reagent_b.SMILES:
                    continue
                reagent_a, reagent_b = sorted([reagent_a, reagent_b], key=lambda x: x.SMILES)
                delta = reagent_b.pIC50 - reagent_a.pIC50
                delta_list.append(list(reagent_a.values) + list(reagent_b.values)
                                  + [f"{reagent_a.R_group.replace('*', '*-')}>>{reagent_b.R_group.replace('*', '*-')}", delta])
        pair_progress.progress((idx + 1) / len(groups))
    
    pair_progress.empty()
    
    cols = ["SMILES_1", "Core_1", "R_group_1", "Name_1", "pIC50_1",
           "SMILES_2", "Core_2", "Rgroup_1", "Name_2", "pIC50_2",
           "Transform", "Delta"]
    delta_df = pd.DataFrame(delta_list, columns=cols)
    progress_bar.progress(70)
    
    # Step 6: Collect frequently occurring pairs
    status_text.text("📊 Step 5/6: Analyzing frequent transforms...")
    mmp_list = []
    for k, v in delta_df.groupby("Transform"):
        if len(v) > min_transform_occurrence:
            mmp_list.append([k, len(v), v.Delta.values])
    
    mmp_df = pd.DataFrame(mmp_list, columns=["Transform", "Count", "Deltas"])
    mmp_df['idx'] = range(0, len(mmp_df))
    mmp_df['mean_delta'] = [x.mean() for x in mmp_df.Deltas]
    mmp_df['rxn_mol'] = mmp_df.Transform.apply(lambda x: AllChem.ReactionFromSmarts(x, useSmiles=True))
    
    # Create index linking dataframes
    transform_dict = dict([(a, b) for a, b in mmp_df[["Transform", "idx"]].values])
    delta_df['idx'] = [transform_dict.get(x) for x in delta_df.Transform]
    
    progress_bar.progress(90)
    
    # Step 7: Display results
    status_text.text("📈 Step 6/6: Preparing results...")
    
    # Sort based on user preference
    mmp_df.sort_values("mean_delta", inplace=True, ascending=show_ascending)
    
    # Display summary statistics
    st.subheader("📋 Analysis Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Molecules", len(df))
    with col2:
        st.metric("Total MMPs Found", len(delta_df))
    with col3:
        st.metric("Unique Transforms", len(mmp_df))
    with col4:
        st.metric("Filtered Transforms", f"> {min_transform_occurrence} occurrences")
    
    progress_bar.progress(100)
    status_text.text("✅ Analysis complete!")
    time.sleep(0.5)
    status_text.empty()
    progress_bar.empty()
    
    # Display MMP results
    st.subheader(f"🧬 Top {rows_to_show} Matched Molecular Pair Transforms")
    st.caption(f"Showing transforms that occur at least {min_transform_occurrence} times, sorted by {'decreasing' if show_ascending else 'increasing'} activity")
    
    # Create columns for display
    display_cols = st.columns(2)
    
    for i in range(min(rows_to_show, len(mmp_df))):
        row = mmp_df.iloc[i]
        col_idx = i % 2
        
        with display_cols[col_idx]:
            # Create expander for each MMP
            with st.expander(f"Transform {row['idx']}: ΔpIC50 = {row['mean_delta']:.2f} (n={row['Count']})", expanded=True):
                # Display reaction image
                rxn_img = rxn_to_pil_image(row['rxn_mol'])
                st.image(rxn_img, caption=row['Transform'], use_column_width=True)
                
                # Create distribution plot
                fig, ax = plt.subplots(figsize=(6, 2))
                plot_delta_distribution(row['Deltas'], ax)
                st.pyplot(fig)
                plt.close()
    
    # Detailed view section
    st.subheader("🔍 Detailed View of Specific MMP")
    
    # Let user select a transform to examine in detail
    if len(mmp_df) > 0:
        transform_options = [f"{idx}: {transform} (Δ={mean_delta:.2f}, n={count})" 
                           for idx, transform, count, mean_delta in 
                           zip(mmp_df['idx'], mmp_df['Transform'], mmp_df['Count'], mmp_df['mean_delta'])]
        
        selected_transform = st.selectbox(
            "Select a transform to view detailed examples:",
            transform_options,
            index=0
        )
        
        # Extract idx from selection
        query_idx = int(selected_transform.split(":")[0])
        
        # Find examples for selected transform
        example_list = []
        for _, row in delta_df.query("idx == @query_idx").sort_values("Delta", ascending=False).iterrows():
            smi_1, name_1, pIC50_1 = row.SMILES_1, row.Name_1, row.pIC50_1
            smi_2, name_2, pIC50_2 = row.SMILES_2, row.Name_2, row.pIC50_2
            tmp_list = [(smi_1, name_1, pIC50_1), (smi_2, name_2, pIC50_2)]
            tmp_list.sort(key=itemgetter(0))
            example_list.append(tmp_list[0])
            example_list.append(tmp_list[1])
        
        example_df = pd.DataFrame(example_list, columns=["SMILES", "Name", "pIC50"])
        
        # Display molecules in a grid
        st.write(f"**Molecules containing transform {query_idx}:**")
        
        # Display as a table with molecule images
        st.dataframe(
            example_df,
            column_config={
                "SMILES": "SMILES",
                "Name": "Name",
                "pIC50": st.column_config.NumberColumn(
                    "pIC50",
                    format="%.2f"
                )
            },
            use_container_width=True,
            hide_index=True
        )
        
        # Show raw data option
        with st.expander("View Raw Data"):
            st.dataframe(mmp_df[['Transform', 'Count', 'mean_delta']].round(3), use_container_width=True)
    
    # Add download buttons
    st.subheader("💾 Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Convert DataFrames to CSV
        csv_mmp = mmp_df[['Transform', 'Count', 'mean_delta']].to_csv(index=False)
        st.download_button(
            label="Download MMP Results (CSV)",
            data=csv_mmp,
            file_name="mmp_results.csv",
            mime="text/csv"
        )
    
    with col2:
        csv_delta = delta_df.to_csv(index=False)
        st.download_button(
            label="Download All Pairs (CSV)",
            data=csv_delta,
            file_name="all_pairs.csv",
            mime="text/csv"
        )
    
    # Add some explanation
    with st.expander("💡 How to interpret the results"):
        st.markdown("""
        **Understanding the Output:**
        
        1. **MMP Transform**: Shows the chemical transformation between two R-groups
        2. **ΔpIC50**: Mean change in pIC50 (negative log of IC50) for this transformation
        3. **Distribution Plot**: Shows individual ΔpIC50 values for each occurrence
        
        **Key Insights:**
        - **Positive ΔpIC50**: Transformation increases activity (lower IC50)
        - **Negative ΔpIC50**: Transformation decreases activity (higher IC50)
        - **Consistent pattern**: Multiple occurrences with similar ΔpIC50 suggest reliable SAR trend
        
        **Example Interpretation:**
        If you see a transform with consistently negative ΔpIC50, it suggests that making
        that chemical change generally reduces hERG inhibition, which is desirable for
        reducing cardiotoxicity risk.
        """)

else:
    # Initial state before analysis
    st.info("👈 Click the 'Start Analysis' button to begin the MMP analysis.")
    
    # Show example of what the app does
    st.subheader("📖 What this app does:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **1. Data Loading**
        - Loads hERG inhibition data from ChEMBL
        - Processes SMILES to RDKit molecules
        - Removes salts and counterions
        """)
    
    with col2:
        st.markdown("""
        **2. MMP Identification**
        - Decomposes molecules into scaffolds & R-groups
        - Finds pairs sharing same scaffold
        - Calculates activity differences (ΔpIC50)
        """)
    
    with col3:
        st.markdown("""
        **3. Analysis & Visualization**
        - Identifies frequent chemical transformations
        - Shows impact on biological activity
        - Visualizes results interactively
        """)
    
    # Show sample data structure
    with st.expander("📊 Sample Data Structure"):
        st.markdown("""
        The dataset contains the following columns:
        
        | Column | Description |
        |--------|-------------|
        | SMILES | Molecular structure as SMILES string |
        | Name | ChEMBL compound identifier |
        | pIC50 | Negative log of IC50 value (-log10(IC50)) |
        """)
        
        # Show a small preview
        sample_data = pd.DataFrame({
            'SMILES': ['CC(=O)Oc1ccccc1C(=O)O', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'],
            'Name': ['CHEMBL25', 'CHEMBL113'],
            'pIC50': [5.2, 6.8]
        })
        st.dataframe(sample_data, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Matched Molecular Pairs Analysis Tool | Built with Streamlit, RDKit, and Pandas"
    "</div>",
    unsafe_allow_html=True
)
