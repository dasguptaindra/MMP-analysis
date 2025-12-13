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
import tempfile
import os

# Set page configuration
st.set_page_config(
    page_title="Matched Molecular Pairs Analysis",
    page_icon="🧪",
    layout="wide"
)

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
    .css-1d391kg {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🧪 Matched Molecular Pairs Analysis")
st.markdown("""
This application performs Matched Molecular Pairs (MMP) analysis to identify the impact of specific chemical changes on biological activity.
""")

# Utility functions
def remove_map_nums(mol):
    """Remove atom map numbers from a molecule"""
    if mol is not None:
        for atm in mol.GetAtoms():
            atm.SetAtomMapNum(0)
    return mol

def sort_fragments(mol):
    """
    Transform a molecule with multiple fragments into a list of molecules 
    that is sorted by number of atoms from largest to smallest
    """
    if mol is None:
        return []
    frag_list = list(Chem.GetMolFrags(mol, asMols=True))
    frag_list = [remove_map_nums(x) for x in frag_list]
    frag_num_atoms_list = [(x.GetNumAtoms(), x) for x in frag_list if x is not None]
    frag_num_atoms_list.sort(key=itemgetter(0), reverse=True)
    return [x[1] for x in frag_num_atoms_list]

def rxn_to_pil_image(rxn):
    """Convert an RDKit reaction to a PIL Image"""
    if rxn is None:
        # Return a blank image
        img = Image.new('RGB', (400, 200), color='white')
        return img
    
    try:
        drawer = rdMolDraw2D.MolDraw2DCairo(400, 200)
        drawer.DrawReaction(rxn)
        drawer.FinishDrawing()
        img_bytes = drawer.GetDrawingText()
        img = Image.open(io.BytesIO(img_bytes))
        return img
    except:
        # Return a blank image on error
        img = Image.new('RGB', (400, 200), color='white')
        return img

def plot_delta_distribution(dist, ax):
    """Plot a distribution as a seaborn stripplot"""
    if len(dist) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12)
        ax.set_xlim(-5, 5)
    else:
        sns.stripplot(x=dist, ax=ax, size=8, alpha=0.7)
        ax.axvline(0, ls="--", c="red", alpha=0.7)
        ax.set_xlim(-5, 5)
    ax.set_xlabel("ΔpIC50", fontsize=10)
    ax.set_title("Distribution of Activity Changes", fontsize=12)
    ax.grid(True, alpha=0.3)
    return ax

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")

# Data source selection
data_source = st.sidebar.radio(
    "Choose Data Source:",
    ["Use Default Dataset", "Upload CSV File"]
)

df = None
uploaded_file = None

if data_source == "Upload CSV File":
    st.sidebar.subheader("📁 Upload Your Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="Upload a CSV file with columns: SMILES, Name, pIC50"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"✅ File uploaded successfully! ({len(df)} rows)")
            
            # Show preview
            with st.sidebar.expander("📊 Data Preview"):
                st.dataframe(df.head(), use_container_width=True)
                
            # Check required columns
            required_cols = ['SMILES', 'Name', 'pIC50']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                st.sidebar.error(f"❌ Missing required columns: {missing_cols}")
                st.sidebar.info("Please ensure your CSV has columns: SMILES, Name, pIC50")
                df = None
            else:
                st.sidebar.success("✅ All required columns found!")
                
        except Exception as e:
            st.sidebar.error(f"❌ Error reading file: {e}")
            df = None
else:
    # Use default dataset
    default_url = "https://raw.githubusercontent.com/PatWalters/practical_cheminformatics_tutorials/main/data/hERG.csv"
    try:
        df = pd.read_csv(default_url)
        st.sidebar.success(f"✅ Default dataset loaded! ({len(df)} rows)")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading default dataset: {e}")

# Parameters section
st.sidebar.subheader("🔧 Analysis Parameters")

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
**Default Dataset Info:**
- Source: ChEMBL hERG inhibition data
- Columns: SMILES, Name, pIC50
- Compounds: ~2,000 molecules
""")

# Main application logic
def run_analysis(df, min_transform_occurrence, rows_to_show, show_ascending):
    """Run the complete MMP analysis"""
    
    # Create containers for progress updates
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 1: Process molecules
    status_text.text("⚗️ Step 1/6: Processing molecules...")
    
    # Validate SMILES
    valid_mask = df['SMILES'].apply(lambda x: Chem.MolFromSmiles(x) is not None)
    valid_df = df[valid_mask].copy()
    
    if len(valid_df) < len(df):
        st.warning(f"⚠️ {len(df) - len(valid_df)} invalid SMILES removed. {len(valid_df)} valid molecules remaining.")
    
    if len(valid_df) == 0:
        st.error("❌ No valid molecules found in the dataset!")
        return None, None
    
    valid_df['mol'] = valid_df.SMILES.apply(Chem.MolFromSmiles)
    progress_bar.progress(20)
    
    # Step 2: Remove salts and counterions
    status_text.text("🧹 Step 2/6: Removing salts and counterions...")
    valid_df['mol'] = valid_df['mol'].apply(lambda x: uru.get_largest_fragment(x) if x else None)
    progress_bar.progress(30)
    
    # Step 3: Decompose molecules to scaffolds and sidechains
    status_text.text("🔬 Step 3/6: Decomposing molecules into scaffolds and sidechains...")
    row_list = []
    
    # Progress tracking for decomposition
    decomposition_container = st.empty()
    decomposition_progress = decomposition_container.progress(0)
    
    total_rows = len(valid_df)
    for i, row in valid_df.iterrows():
        mol = row['mol']
        if mol is not None:
            frag_list = FragmentMol(mol, maxCuts=1)
            for _, frag_mol in frag_list:
                if frag_mol is not None:
                    pair_list = sort_fragments(frag_mol)
                    if len(pair_list) == 2:  # We need exactly 2 fragments
                        tmp_list = [
                            row['SMILES'],
                            Chem.MolToSmiles(pair_list[0]) if pair_list[0] else "",
                            Chem.MolToSmiles(pair_list[1]) if pair_list[1] else "",
                            row['Name'],
                            row['pIC50']
                        ]
                        row_list.append(tmp_list)
        decomposition_progress.progress((i + 1) / total_rows)
    
    decomposition_container.empty()
    
    if len(row_list) == 0:
        st.error("❌ No valid fragments found! Unable to perform MMP analysis.")
        return None, None
    
    row_df = pd.DataFrame(row_list, columns=["SMILES", "Core", "R_group", "Name", "pIC50"])
    
    # Clean up R_group strings
    row_df['R_group'] = row_df['R_group'].fillna('').astype(str)
    row_df = row_df[row_df['R_group'] != '']
    
    progress_bar.progress(50)
    
    # Step 4: Collect pairs with same scaffold
    status_text.text("🔗 Step 4/6: Finding matched molecular pairs...")
    delta_list = []
    
    # Group by core and find pairs
    groups = list(row_df.groupby("Core"))
    pair_container = st.empty()
    pair_progress = pair_container.progress(0)
    
    for idx, (k, v) in enumerate(groups):
        if len(v) > 2:
            for a, b in combinations(range(0, len(v)), 2):
                reagent_a = v.iloc[a]
                reagent_b = v.iloc[b]
                if reagent_a.SMILES == reagent_b.SMILES:
                    continue
                
                # Sort by SMILES for consistency
                if reagent_a.SMILES > reagent_b.SMILES:
                    reagent_a, reagent_b = reagent_b, reagent_a
                
                # Ensure R_groups are strings
                r_group_a = str(reagent_a.R_group).replace('*', '*-')
                r_group_b = str(reagent_b.R_group).replace('*', '*-')
                
                delta = float(reagent_b.pIC50) - float(reagent_a.pIC50)
                delta_list.append([
                    reagent_a.SMILES, reagent_a.Core, reagent_a.R_group, 
                    reagent_a.Name, float(reagent_a.pIC50),
                    reagent_b.SMILES, reagent_b.Core, reagent_b.R_group,
                    reagent_b.Name, float(reagent_b.pIC50),
                    f"{r_group_a}>>{r_group_b}", delta
                ])
        pair_progress.progress((idx + 1) / len(groups))
    
    pair_container.empty()
    
    if len(delta_list) == 0:
        st.error("❌ No matched molecular pairs found!")
        return None, None
    
    cols = ["SMILES_1", "Core_1", "R_group_1", "Name_1", "pIC50_1",
           "SMILES_2", "Core_2", "Rgroup_1", "Name_2", "pIC50_2",
           "Transform", "Delta"]
    delta_df = pd.DataFrame(delta_list, columns=cols)
    progress_bar.progress(70)
    
    # Step 5: Collect frequently occurring pairs
    status_text.text("📊 Step 5/6: Analyzing frequent transforms...")
    mmp_list = []
    
    for k, v in delta_df.groupby("Transform"):
        if len(v) > min_transform_occurrence:
            # Convert to list for storage
            deltas = v['Delta'].tolist()
            mmp_list.append([k, len(v), deltas])
    
    if len(mmp_list) == 0:
        st.warning(f"⚠️ No transforms found with occurrence > {min_transform_occurrence}. Try lowering the threshold.")
        return None, None
    
    mmp_df = pd.DataFrame(mmp_list, columns=["Transform", "Count", "Deltas"])
    mmp_df['idx'] = range(0, len(mmp_df))
    mmp_df['mean_delta'] = [sum(x)/len(x) if len(x) > 0 else 0 for x in mmp_df.Deltas]
    
    # Create reaction molecules (handle errors)
    def create_reaction(transform_str):
        try:
            # Clean up the transform string
            clean_str = transform_str.replace('*-', '*')
            return AllChem.ReactionFromSmarts(clean_str, useSmiles=True)
        except:
            return None
    
    mmp_df['rxn_mol'] = mmp_df['Transform'].apply(create_reaction)
    
    # Create index linking dataframes
    transform_dict = dict([(a, b) for a, b in mmp_df[["Transform", "idx"]].values])
    delta_df['idx'] = [transform_dict.get(x, -1) for x in delta_df.Transform]
    
    progress_bar.progress(90)
    
    # Step 6: Finalize
    status_text.text("📈 Step 6/6: Preparing results...")
    progress_bar.progress(100)
    time.sleep(0.5)
    status_text.empty()
    progress_bar.empty()
    
    return mmp_df, delta_df

# Main application flow
if df is not None:
    # Display dataset info
    st.subheader("📋 Dataset Information")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Compounds", len(df))
    with col2:
        st.metric("pIC50 Range", f"{df['pIC50'].min():.1f} - {df['pIC50'].max():.1f}")
    with col3:
        st.metric("Unique Names", df['Name'].nunique())
    
    with st.expander("🔍 View Dataset"):
        st.dataframe(df.head(20), use_container_width=True)
    
    # Run analysis button
    if st.button("🚀 Start MMP Analysis", type="primary", use_container_width=True):
        
        # Run the analysis
        with st.spinner("Running analysis... This may take a few minutes."):
            mmp_df, delta_df = run_analysis(df, min_transform_occurrence, rows_to_show, show_ascending)
        
        if mmp_df is not None and delta_df is not None:
            # Display results
            st.success(f"✅ Analysis complete! Found {len(mmp_df)} frequent transforms.")
            
            # Display summary statistics
            st.subheader("📊 Analysis Results")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Frequent Transforms", len(mmp_df))
            with col2:
                avg_delta = mmp_df['mean_delta'].mean()
                st.metric("Avg ΔpIC50", f"{avg_delta:.2f}")
            with col3:
                max_delta = mmp_df['mean_delta'].max()
                st.metric("Max ΔpIC50", f"{max_delta:.2f}")
            with col4:
                min_delta = mmp_df['mean_delta'].min()
                st.metric("Min ΔpIC50", f"{min_delta:.2f}")
            
            # Sort based on user preference
            mmp_df_sorted = mmp_df.sort_values("mean_delta", ascending=show_ascending).copy()
            
            # Display MMP results
            st.subheader(f"🧬 Top {min(rows_to_show, len(mmp_df_sorted))} Matched Molecular Pair Transforms")
            
            if show_ascending:
                st.caption(f"Showing transforms that decrease activity first (most negative ΔpIC50)")
            else:
                st.caption(f"Showing transforms that increase activity first (most positive ΔpIC50)")
            
            # Create columns for display
            display_cols = st.columns(2)
            
            for i in range(min(rows_to_show, len(mmp_df_sorted))):
                row = mmp_df_sorted.iloc[i]
                col_idx = i % 2
                
                with display_cols[col_idx]:
                    # Create card for each MMP
                    with st.container():
                        st.markdown(f"**Transform {row['idx']}**")
                        
                        # Create columns within the card
                        card_col1, card_col2 = st.columns([2, 1])
                        
                        with card_col1:
                            # Display reaction if available
                            if row['rxn_mol'] is not None:
                                rxn_img = rxn_to_pil_image(row['rxn_mol'])
                                st.image(rxn_img, use_column_width=True)
                            else:
                                st.code(row['Transform'])
                            
                            st.caption(f"Occurrences: {row['Count']}")
                        
                        with card_col2:
                            # Display metrics
                            st.metric("ΔpIC50", f"{row['mean_delta']:.2f}")
                            
                            # Create small distribution plot
                            fig, ax = plt.subplots(figsize=(3, 1.5))
                            if len(row['Deltas']) > 0:
                                ax.boxplot(row['Deltas'], vert=False, widths=0.6)
                                ax.scatter(row['Deltas'], [1]*len(row['Deltas']), alpha=0.5, s=20)
                            ax.axvline(0, color='red', linestyle='--', alpha=0.5)
                            ax.set_xlim(-5, 5)
                            ax.set_xlabel('ΔpIC50')
                            ax.set_yticks([])
                            st.pyplot(fig, use_container_width=True)
                            plt.close()
                        
                        st.markdown("---")
            
            # Detailed view section
            st.subheader("🔍 Explore Specific Transform")
            
            if len(mmp_df_sorted) > 0:
                # Create selector for transforms
                transform_options = [
                    f"{idx}: {transform[:50]}{'...' if len(transform) > 50 else ''} "
                    f"(Δ={mean_delta:.2f}, n={count})"
                    for idx, transform, count, mean_delta in 
                    zip(mmp_df_sorted['idx'], mmp_df_sorted['Transform'], 
                        mmp_df_sorted['Count'], mmp_df_sorted['mean_delta'])
                ]
                
                selected_idx = st.selectbox(
                    "Select a transform to view detailed examples:",
                    range(len(mmp_df_sorted)),
                    format_func=lambda x: transform_options[x]
                )
                
                if selected_idx is not None:
                    query_idx = mmp_df_sorted.iloc[selected_idx]['idx']
                    
                    # Get examples for selected transform
                    examples = delta_df[delta_df['idx'] == query_idx].copy()
                    
                    if len(examples) > 0:
                        # Display transform info
                        st.markdown(f"**Selected Transform:** `{mmp_df_sorted.iloc[selected_idx]['Transform']}`")
                        
                        # Show examples in a table
                        example_data = []
                        for _, row in examples.iterrows():
                            example_data.append({
                                'Molecule 1 SMILES': row['SMILES_1'],
                                'Molecule 1 Name': row['Name_1'],
                                'Molecule 1 pIC50': f"{row['pIC50_1']:.2f}",
                                'Molecule 2 SMILES': row['SMILES_2'],
                                'Molecule 2 Name': row['Name_2'],
                                'Molecule 2 pIC50': f"{row['pIC50_2']:.2f}",
                                'ΔpIC50': f"{row['Delta']:.2f}"
                            })
                        
                        example_df = pd.DataFrame(example_data)
                        
                        # Display with alternating row colors
                        st.dataframe(
                            example_df.style.apply(
                                lambda x: ['background: #f0f2f6' if i%2==0 else '' for i in range(len(x))],
                                axis=0
                            ),
                            use_container_width=True,
                            height=min(400, len(examples) * 35 + 38)
                        )
                    else:
                        st.info("No examples found for this transform.")
            
            # Data export section
            st.subheader("💾 Export Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📥 Download MMP Summary", use_container_width=True):
                    csv = mmp_df_sorted[['Transform', 'Count', 'mean_delta']].to_csv(index=False)
                    st.download_button(
                        label="Click to Download",
                        data=csv,
                        file_name="mmp_summary.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            with col2:
                if st.button("📥 Download All Pairs", use_container_width=True):
                    csv = delta_df.to_csv(index=False)
                    st.download_button(
                        label="Click to Download",
                        data=csv,
                        file_name="all_pairs.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            with col3:
                if st.button("📥 Download Full Results", use_container_width=True):
                    # Combine all data
                    full_results = pd.merge(
                        delta_df,
                        mmp_df_sorted[['Transform', 'mean_delta']],
                        on='Transform',
                        how='left',
                        suffixes=('', '_mean')
                    )
                    csv = full_results.to_csv(index=False)
                    st.download_button(
                        label="Click to Download",
                        data=csv,
                        file_name="full_mmp_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            # Interpretation guide
            with st.expander("📚 Interpretation Guide"):
                st.markdown("""
                ### Understanding the Results
                
                **Key Metrics:**
                - **ΔpIC50**: Mean change in activity for the transform
                - **Positive ΔpIC50**: Activity increases (lower IC50, more potent)
                - **Negative ΔpIC50**: Activity decreases (higher IC50, less potent)
                - **Count**: Number of times this transform was observed
                
                **For hERG Inhibition Analysis:**
                - Negative ΔpIC50 is generally desirable (reduces hERG inhibition, lower cardiotoxicity risk)
                - Positive ΔpIC50 indicates increased hERG inhibition (potential safety concern)
                
                **Example:** If changing "chlorobenzyl" to "benzyl" gives ΔpIC50 = -1.5, 
                this means the change typically reduces hERG inhibition by 1.5 log units.
                """)
                
else:
    # Initial state - show instructions
    st.info("👈 Please select a data source in the sidebar to begin.")
    
    # Show example of expected format
    with st.expander("📋 Expected Data Format"):
        st.markdown("""
        Your CSV file should have the following columns:
        
        | Column | Required | Description | Example |
        |--------|----------|-------------|---------|
        | SMILES | Yes | Molecular structure as SMILES string | `CC(=O)Oc1ccccc1C(=O)O` |
        | Name | Yes | Compound identifier | `CHEMBL25` |
        | pIC50 | Yes | Negative log of IC50 value | `5.2` |
        
        **Example CSV content:**
        ```csv
        SMILES,Name,pIC50
        CC(=O)Oc1ccccc1C(=O)O,CHEMBL25,5.2
        CN1C=NC2=C1C(=O)N(C(=O)N2C)C,CHEMBL113,6.8
        C1=CC=C(C=C1)C=O,CHEMBL196,4.5
        ```
        
        **Note:** The default dataset contains ~2,000 compounds from ChEMBL's hERG inhibition assay.
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Matched Molecular Pairs Analysis Tool | Built with Streamlit, RDKit, and Pandas | "
    "<a href='https://doi.org/10.1021/ci900450m' target='_blank'>Original MMP Method</a>"
    "</div>",
    unsafe_allow_html=True
)
