import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdMolDraw2D
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import BytesIO
import base64
from PIL import Image
import io
from itertools import combinations
from operator import itemgetter
import tempfile
import os

# Set page config
st.set_page_config(
    page_title="MMP Analysis",
    page_icon="🧪",
    layout="wide"
)

# Title
st.title("🧪 Matched Molecular Pair (MMP) Analysis")
st.markdown("""
This app performs Matched Molecular Pair analysis to identify structural transformations 
that influence compound potency (pIC50 values).
""")

# Sidebar for file upload and parameters
st.sidebar.header("📁 Data Input")

uploaded_file = st.sidebar.file_uploader(
    "Upload your CSV file with columns: SMILES, Name, pIC50",
    type=['csv']
)

# Example data in sidebar
st.sidebar.markdown("### 📋 Expected CSV Format")
st.sidebar.code("""SMILES,Name,pIC50
Cc1ccc(cc1)S(=O)(=O)N,Compound1,7.2
COc1ccc(cc1)S(=O)(=O)N,Compound2,7.8
...""")

# Parameters in sidebar
st.sidebar.header("⚙️ Parameters")
min_transform_occurrence = st.sidebar.slider(
    "Minimum transform occurrences",
    min_value=1,
    max_value=20,
    value=5,
    help="Only consider transformations that occur at least this many times"
)

num_top_transforms = st.sidebar.slider(
    "Number of top transforms to display",
    min_value=3,
    max_value=10,
    value=3
)

# Function definitions
def remove_map_nums(mol):
    """Remove atom map numbers from a molecule"""
    for atm in mol.GetAtoms():
        atm.SetAtomMapNum(0)
    return mol

def sort_fragments(mol):
    """
    Transform a molecule with multiple fragments into a list of molecules 
    that is sorted by number of atoms from largest to smallest
    """
    frag_list = list(Chem.GetMolFrags(mol, asMols=True))
    frag_list = [remove_map_nums(x) for x in frag_list]
    frag_num_atoms_list = [(x.GetNumAtoms(), x) for x in frag_list]
    frag_num_atoms_list.sort(key=itemgetter(0), reverse=True)
    return [x[1] for x in frag_num_atoms_list]

# Scaffold finder function (simplified version)
def FragmentMol(mol, maxCuts=1):
    """Simplified fragmentation function for MMP analysis"""
    from rdkit.Chem.rdMMPA import FragmentMol as RDKitFragmentMol
    return RDKitFragmentMol(mol, maxCuts=maxCuts)

def get_largest_fragment(mol):
    """Get the largest fragment from a molecule"""
    frags = Chem.GetMolFrags(mol, asMols=True)
    if frags:
        return max(frags, key=lambda x: x.GetNumAtoms())
    return mol

# Plotting functions
def rxn_to_image(rxn_mol, width=300, height=150):
    """Convert RDKit reaction to PIL Image"""
    img = Draw.ReactionToImage(rxn_mol, subImgSize=(width, height))
    return img

def mol_to_image(mol, width=200, height=200):
    """Convert RDKit molecule to PIL Image"""
    img = Draw.MolToImage(mol, size=(width, height))
    return img

def create_stripplot(deltas, figsize=(4, 1.5)):
    """Create a stripplot for delta distribution"""
    fig, ax = plt.subplots(figsize=figsize)
    sns.stripplot(x=deltas, ax=ax, jitter=0.3, alpha=0.7, s=8, color='steelblue')
    ax.axvline(0, ls="--", c="red", alpha=0.7)
    ax.set_xlim(-5, 5)
    ax.set_xlabel("ΔpIC50")
    ax.set_ylabel("")
    ax.set_yticks([])
    plt.tight_layout()
    return fig

def display_compound_grid(compounds_df, smiles_col="SMILES", id_col="Name", value_col="pIC50"):
    """Display compounds in a grid format"""
    n_cols = 4
    n_rows = (len(compounds_df) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    if n_rows == 1:
        axes = [axes]
    if n_cols == 1:
        axes = [[ax] for ax in axes]
    
    for idx, (_, row) in enumerate(compounds_df.iterrows()):
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        
        mol = Chem.MolFromSmiles(row[smiles_col])
        if mol:
            img = Draw.MolToImage(mol, size=(250, 250))
            axes[row_idx][col_idx].imshow(img)
            title = f"{row[id_col]}\n{value_col}: {row[value_col]:.2f}"
            axes[row_idx][col_idx].set_title(title, fontsize=9)
        else:
            axes[row_idx][col_idx].text(0.5, 0.5, "Invalid SMILES", 
                                       ha='center', va='center')
        
        axes[row_idx][col_idx].axis('off')
    
    # Hide empty subplots
    for idx in range(len(compounds_df), n_rows * n_cols):
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        axes[row_idx][col_idx].axis('off')
    
    plt.tight_layout()
    return fig

# Main analysis function
def run_mmp_analysis(df, min_occurrences=5):
    """Main MMP analysis pipeline"""
    
    # Process molecules
    df['mol'] = df['SMILES'].apply(Chem.MolFromSmiles)
    df['mol'] = df['mol'].apply(get_largest_fragment)
    
    # Decompose molecules to scaffolds and sidechains
    st.info("🔬 Decomposing molecules...")
    progress_bar = st.progress(0)
    
    row_list = []
    for idx, (smiles, name, pIC50, mol) in enumerate(df[['SMILES', 'Name', 'pIC50', 'mol']].values):
        frag_list = FragmentMol(mol, maxCuts=1)
        for _, frag_mol in frag_list:
            pair_list = sort_fragments(frag_mol)
            tmp_list = [smiles] + [Chem.MolToSmiles(x) for x in pair_list] + [name, pIC50]
            row_list.append(tmp_list)
        progress_bar.progress((idx + 1) / len(df))
    
    row_df = pd.DataFrame(row_list, columns=["SMILES", "Core", "R_group", "Name", "pIC50"])
    
    # Collect pairs with same scaffold
    st.info("🤝 Finding molecular pairs...")
    delta_list = []
    
    for k, v in row_df.groupby("Core"):
        if len(v) > 2:
            for a, b in combinations(range(len(v)), 2):
                reagent_a = v.iloc[a]
                reagent_b = v.iloc[b]
                if reagent_a.SMILES == reagent_b.SMILES:
                    continue
                reagent_a, reagent_b = sorted([reagent_a, reagent_b], key=lambda x: x.SMILES)
                delta = reagent_b.pIC50 - reagent_a.pIC50
                delta_list.append(list(reagent_a.values) + list(reagent_b.values) +
                                 [f"{reagent_a.R_group.replace('*', '*-')}>>{reagent_b.R_group.replace('*', '*-')}", delta])
    
    cols = ["SMILES_1", "Core_1", "R_group_1", "Name_1", "pIC50_1",
           "SMILES_2", "Core_2", "Rgroup_1", "Name_2", "pIC50_2",
           "Transform", "Delta"]
    delta_df = pd.DataFrame(delta_list, columns=cols)
    
    # Collect frequently occurring pairs
    st.info("📊 Analyzing transformations...")
    mmp_list = []
    for k, v in delta_df.groupby("Transform"):
        if len(v) > min_occurrences:
            mmp_list.append([k, len(v), v.Delta.values])
    
    mmp_df = pd.DataFrame(mmp_list, columns=["Transform", "Count", "Deltas"])
    mmp_df['idx'] = range(len(mmp_df))
    mmp_df['mean_delta'] = [x.mean() for x in mmp_df.Deltas]
    mmp_df['rxn_mol'] = mmp_df.Transform.apply(
        lambda x: AllChem.ReactionFromSmarts(x.replace('*-', '*'), useSmiles=True) if x else None
    )
    
    return df, row_df, delta_df, mmp_df

# Main app logic
if uploaded_file is not None:
    try:
        # Load data
        df = pd.read_csv(uploaded_file)
        
        # Check required columns
        required_cols = ['SMILES', 'Name', 'pIC50']
        if not all(col in df.columns for col in required_cols):
            st.error(f"CSV file must contain columns: {required_cols}")
            st.stop()
        
        # Display data preview
        st.header("📊 Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        st.write(f"Total compounds: {len(df)}")
        
        # Run MMP analysis
        with st.spinner("Running MMP analysis..."):
            df_processed, row_df, delta_df, mmp_df = run_mmp_analysis(df, min_transform_occurrence)
        
        st.success(f"✅ Analysis complete! Found {len(mmp_df)} significant transformations")
        
        # Display results in tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Top Transforms", "📉 Bottom Transforms", "🔍 All Transforms", "📥 Export"])
        
        with tab1:
            st.header("🏆 Top Potency-Enhancing Transformations")
            
            if len(mmp_df) > 0:
                # Sort by mean_delta descending for positive effects
                top_positive = mmp_df.sort_values('mean_delta', ascending=False).head(num_top_transforms)
                
                for i, (idx, row) in enumerate(top_positive.iterrows()):
                    col1, col2, col3 = st.columns([2, 2, 3])
                    
                    with col1:
                        st.subheader(f"Rank #{i+1}")
                        st.write(f"**Transform:** `{row['Transform']}`")
                        st.write(f"**Mean ΔpIC50:** {row['mean_delta']:.3f}")
                        st.write(f"**Occurrences:** {row['Count']}")
                    
                    with col2:
                        # Display reaction
                        if row['rxn_mol']:
                            img = rxn_to_image(row['rxn_mol'], width=400, height=200)
                            st.image(img, caption="Reaction Transform")
                    
                    with col3:
                        # Display stripplot
                        fig = create_stripplot(row['Deltas'], figsize=(4, 2))
                        st.pyplot(fig)
                    
                    # Show example compounds for this transform
                    transform_idx = row['idx']
                    compound_pairs = delta_df[delta_df['idx'] == transform_idx].head(3)
                    
                    if not compound_pairs.empty:
                        st.markdown("**Example Compound Pairs:**")
                        examples_data = []
                        for _, pair in compound_pairs.iterrows():
                            examples_data.extend([
                                {'SMILES': pair['SMILES_1'], 'Name': pair['Name_1'], 'pIC50': pair['pIC50_1']},
                                {'SMILES': pair['SMILES_2'], 'Name': pair['Name_2'], 'pIC50': pair['pIC50_2']}
                            ])
                        
                        examples_df = pd.DataFrame(examples_data)
                        fig = display_compound_grid(examples_df)
                        st.pyplot(fig)
                    
                    st.markdown("---")
            else:
                st.warning("No significant transformations found. Try lowering the minimum occurrence threshold.")
        
        with tab2:
            st.header("⚠️ Top Potency-Diminishing Transformations")
            
            if len(mmp_df) > 0:
                # Sort by mean_delta ascending for negative effects
                top_negative = mmp_df.sort_values('mean_delta', ascending=True).head(num_top_transforms)
                
                for i, (idx, row) in enumerate(top_negative.iterrows()):
                    col1, col2, col3 = st.columns([2, 2, 3])
                    
                    with col1:
                        st.subheader(f"Rank #{i+1}")
                        st.write(f"**Transform:** `{row['Transform']}`")
                        st.write(f"**Mean ΔpIC50:** {row['mean_delta']:.3f}")
                        st.write(f"**Occurrences:** {row['Count']}")
                    
                    with col2:
                        # Display reaction
                        if row['rxn_mol']:
                            img = rxn_to_image(row['rxn_mol'], width=400, height=200)
                            st.image(img, caption="Reaction Transform")
                    
                    with col3:
                        # Display stripplot
                        fig = create_stripplot(row['Deltas'], figsize=(4, 2))
                        st.pyplot(fig)
                    
                    # Show example compounds for this transform
                    transform_idx = row['idx']
                    compound_pairs = delta_df[delta_df['idx'] == transform_idx].head(3)
                    
                    if not compound_pairs.empty:
                        st.markdown("**Example Compound Pairs:**")
                        examples_data = []
                        for _, pair in compound_pairs.iterrows():
                            examples_data.extend([
                                {'SMILES': pair['SMILES_1'], 'Name': pair['Name_1'], 'pIC50': pair['pIC50_1']},
                                {'SMILES': pair['SMILES_2'], 'Name': pair['Name_2'], 'pIC50': pair['pIC50_2']}
                            ])
                        
                        examples_df = pd.DataFrame(examples_data)
                        fig = display_compound_grid(examples_df)
                        st.pyplot(fig)
                    
                    st.markdown("---")
        
        with tab3:
            st.header("📋 All Significant Transformations")
            
            if len(mmp_df) > 0:
                # Sort all transforms by mean_delta
                mmp_df_sorted = mmp_df.sort_values('mean_delta', ascending=False)
                
                # Display as table
                display_df = mmp_df_sorted[['Transform', 'Count', 'mean_delta']].copy()
                display_df['mean_delta'] = display_df['mean_delta'].round(3)
                display_df.columns = ['Transform', 'Occurrences', 'Mean ΔpIC50']
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    column_config={
                        "Transform": st.column_config.TextColumn("Transform", width="large"),
                        "Occurrences": st.column_config.NumberColumn("Occurrences", width="small"),
                        "Mean ΔpIC50": st.column_config.NumberColumn("Mean ΔpIC50", width="small",
                                                                     format="%.3f")
                    }
                )
                
                # Distribution plot of all mean deltas
                st.subheader("Distribution of Transformation Effects")
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.hist(mmp_df_sorted['mean_delta'], bins=20, alpha=0.7, color='steelblue', edgecolor='black')
                ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
                ax.set_xlabel('Mean ΔpIC50')
                ax.set_ylabel('Frequency')
                ax.set_title('Distribution of Transformation Effects')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            else:
                st.warning("No significant transformations found.")
        
        with tab4:
            st.header("💾 Export Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Export mmp_df
                csv1 = mmp_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Transformations (CSV)",
                    data=csv1,
                    file_name="mmp_transformations.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Export delta_df
                csv2 = delta_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download All Pairs (CSV)",
                    data=csv2,
                    file_name="mmp_pairs.csv",
                    mime="text/csv"
                )
            
            with col3:
                # Export summary statistics
                summary_stats = {
                    'Total Compounds': len(df),
                    'Total Pairs': len(delta_df),
                    'Significant Transforms': len(mmp_df),
                    'Avg Mean ΔpIC50': mmp_df['mean_delta'].mean() if len(mmp_df) > 0 else 0,
                    'Most Positive Transform': mmp_df.loc[mmp_df['mean_delta'].idxmax(), 'Transform'] if len(mmp_df) > 0 else 'N/A',
                    'Most Negative Transform': mmp_df.loc[mmp_df['mean_delta'].idxmin(), 'Transform'] if len(mmp_df) > 0 else 'N/A'
                }
                
                summary_df = pd.DataFrame(list(summary_stats.items()), columns=['Metric', 'Value'])
                csv3 = summary_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Summary (CSV)",
                    data=csv3,
                    file_name="mmp_summary.csv",
                    mime="text/csv"
                )
            
            # Display summary
            st.subheader("Summary Statistics")
            st.json(summary_stats)
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)

else:
    # Show instructions when no file is uploaded
    st.info("👈 Please upload a CSV file to begin analysis")
    
    # Example section
    st.header("🎯 How to Use This App")
    
    st.markdown("""
    1. **Prepare your data** in a CSV file with columns: `SMILES`, `Name`, `pIC50`
    2. **Upload the file** using the file uploader in the sidebar
    3. **Adjust parameters** (optional):
       - Minimum transform occurrences
       - Number of top transforms to display
    4. **Explore results** in the different tabs:
       - 📈 Top Transforms: Most potency-enhancing transformations
       - 📉 Bottom Transforms: Most potency-diminishing transformations
       - 📋 All Transforms: Complete list of significant transformations
       - 📥 Export: Download results for further analysis
    """)
    
    # Quick demo with example data
    st.header("🚀 Quick Demo")
    if st.button("Run with example data"):
        # Create example data
        example_data = {
            'SMILES': [
                'Cc1ccc(cc1)S(=O)(=O)N',
                'COc1ccc(cc1)S(=O)(=O)N',
                'CFc1ccc(cc1)S(=O)(=O)N',
                'CCc1ccc(cc1)S(=O)(=O)N',
                'CCOc1ccc(cc1)S(=O)(=O)N',
                'CNc1ccc(cc1)S(=O)(=O)N',
            ],
            'Name': ['Compound1', 'Compound2', 'Compound3', 'Compound4', 'Compound5', 'Compound6'],
            'pIC50': [7.2, 7.8, 6.9, 7.5, 8.1, 6.5]
        }
        
        df_example = pd.DataFrame(example_data)
        
        # Save to temp file and run analysis
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df_example.to_csv(f.name, index=False)
            uploaded_file = f.name
        
        st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit & RDKit | Matched Molecular Pair Analysis</p>
    </div>
    """,
    unsafe_allow_html=True
)
