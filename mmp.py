import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdMMPA import FragmentMol
from rdkit.Chem import AllChem, Draw, rdMolDraw2D
from rdkit.Chem.Draw import IPythonConsole
import useful_rdkit_utils as uru
from operator import itemgetter
from itertools import combinations
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import sys
import requests
import tempfile
import os

# Page configuration
st.set_page_config(
    page_title="MMP Analysis Tool",
    page_icon="🧪",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .stDataFrame {
        font-size: 0.9rem;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🧪 MMP (Matched Molecular Pair) Analysis</h1>', unsafe_allow_html=True)

# Sidebar for parameters
with st.sidebar:
    st.header("⚙️ Analysis Parameters")
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    # Analysis parameters
    st.subheader("MMP Settings")
    min_transform_occurrence = st.slider(
        "Minimum transform occurrence",
        min_value=1,
        max_value=20,
        value=5,
        help="Minimum number of times a transform must appear to be included"
    )
    
    # Display options
    st.subheader("Display Options")
    rows_to_show = st.slider("Rows to display", 10, 200, 120)
    sort_ascending = st.checkbox("Sort by ascending delta", value=True)
    show_raw_data = st.checkbox("Show raw data tables", value=False)
    
    # Process button
    process_button = st.button("🚀 Run MMP Analysis", type="primary")

# Helper functions
def remove_map_nums(mol):
    """Remove atom map numbers from a molecule"""
    for atm in mol.GetAtoms():
        atm.SetAtomMapNum(0)
    return mol

def sort_fragments(mol):
    """Transform a molecule with multiple fragments into a list of molecules sorted by number of atoms"""
    frag_list = list(Chem.GetMolFrags(mol, asMols=True))
    [remove_map_nums(x) for x in frag_list]
    frag_num_atoms_list = [(x.GetNumAtoms(), x) for x in frag_list]
    frag_num_atoms_list.sort(key=itemgetter(0), reverse=True)
    return [x[1] for x in frag_num_atoms_list]

def mol_to_image_base64(mol, size=(300, 200)):
    """Convert RDKit molecule to base64 encoded image"""
    try:
        drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        
        # Get PNG data
        png_data = drawer.GetDrawingText()
        
        # Convert to base64
        b64 = base64.b64encode(png_data).decode('utf-8')
        return f'<img src="data:image/png;base64,{b64}" width="{size[0]}" height="{size[1]}">'
    except:
        return "Error generating image"

def rxn_to_base64_image(rxn):
    """Convert reaction to base64 image"""
    try:
        img = Draw.ReactionToImage(rxn, subImgSize=(200, 150))
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        return f'<img src="data:image/png;base64,{img_str}" width="400" height="150">'
    except:
        return "Error generating reaction image"

def stripplot_base64_image(deltas):
    """Create stripplot for delta distribution"""
    try:
        fig, ax = plt.subplots(figsize=(3, 1))
        ax.scatter(deltas, [0] * len(deltas), alpha=0.6, s=50)
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        ax.set_xlim(min(deltas) - 0.5, max(deltas) + 0.5)
        ax.set_yticks([])
        ax.set_xlabel('ΔpIC50')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='PNG', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        return f'<img src="data:image/png;base64,{img_str}" width="300" height="100">'
    except:
        return "Error generating plot"

def load_data(df):
    """Process the input dataframe"""
    # Ensure required columns exist
    required_cols = ['SMILES', 'Name', 'pIC50']
    
    if all(col in df.columns for col in required_cols):
        # Clean data
        df['mol'] = df['SMILES'].apply(Chem.MolFromSmiles)
        df['mol'] = df['mol'].apply(uru.get_largest_fragment)
        df['SMILES'] = df['mol'].apply(Chem.MolToSmiles)
        return df
    else:
        st.error(f"CSV must contain columns: {', '.join(required_cols)}")
        return None

def generate_mmps(df, progress_bar):
    """Generate MMPs from dataframe"""
    row_list = []
    
    # Generate fragments
    progress_bar.progress(0.1, text="Generating molecular fragments...")
    
    for i, (smiles, name, pIC50, mol) in enumerate(df[["SMILES", "Name", "pIC50", "mol"]].values):
        frag_list = FragmentMol(mol, maxCuts=1)
        for _, frag_mol in frag_list:
            if frag_mol:
                pair_list = sort_fragments(frag_mol)
                if len(pair_list) == 2:  # We need exactly two fragments
                    tmp_list = [smiles] + [Chem.MolToSmiles(x) for x in pair_list] + [name, pIC50]
                    row_list.append(tmp_list)
    
    if not row_list:
        st.error("No valid fragments generated. Check your input molecules.")
        return None, None
    
    # Create fragment dataframe
    row_df = pd.DataFrame(row_list, columns=["SMILES", "Core", "R_group", "Name", "pIC50"])
    
    # Generate deltas
    progress_bar.progress(0.5, text="Calculating ΔpIC50 values...")
    
    delta_list = []
    cores = row_df.groupby("Core")
    total_cores = len(cores)
    
    for idx, (k, v) in enumerate(cores):
        if len(v) > 2:
            for a, b in combinations(range(0, len(v)), 2):
                reagent_a = v.iloc[a]
                reagent_b = v.iloc[b]
                if reagent_a.SMILES == reagent_b.SMILES:
                    continue
                
                # Sort to ensure consistent ordering
                reagent_a, reagent_b = sorted([reagent_a, reagent_b], key=lambda x: x.SMILES)
                delta = reagent_b.pIC50 - reagent_a.pIC50
                delta_list.append(list(reagent_a.values) + list(reagent_b.values)
                                  + [f"{reagent_a.R_group.replace('*', '*-')}>>{reagent_b.R_group.replace('*', '*-')}", delta])
        
        # Update progress
        if total_cores > 0:
            progress_bar.progress(0.5 + (idx / total_cores * 0.4))
    
    if not delta_list:
        st.error("No valid MMP pairs found.")
        return None, None
    
    # Create delta dataframe
    cols = ["SMILES_1", "Core_1", "R_group_1", "Name_1", "pIC50_1",
            "SMILES_2", "Core_2", "R_group_2", "Name_2", "pIC50_2",
            "Transform", "Delta"]
    delta_df = pd.DataFrame(delta_list, columns=cols)
    
    # Filter transforms by occurrence
    progress_bar.progress(0.9, text="Filtering transforms...")
    
    mmp_list = []
    for k, v in delta_df.groupby("Transform"):
        if len(v) > min_transform_occurrence:
            mmp_list.append([k, len(v), v.Delta.values])
    
    mmp_df = pd.DataFrame(mmp_list, columns=["Transform", "Count", "Deltas"])
    mmp_df['idx'] = range(0, len(mmp_df))
    mmp_df['mean_delta'] = [x.mean() for x in mmp_df.Deltas]
    
    # Create reaction molecules
    mmp_df['rxn_mol'] = mmp_df.Transform.apply(
        lambda x: AllChem.ReactionFromSmarts(x.replace('*-', '*'), useSmiles=True) if pd.notnull(x) else None
    )
    
    # Create transform dictionary
    transform_dict = dict([(a, b) for a, b in mmp_df[["Transform", "idx"]].values])
    delta_df['idx'] = [transform_dict.get(x) for x in delta_df.Transform]
    
    progress_bar.progress(1.0, text="Analysis complete!")
    
    return row_df, delta_df, mmp_df

# Main app logic
if process_button and uploaded_file is not None:
    # Read and process data
    try:
        df = pd.read_csv(uploaded_file)
        
        # Check if dataframe has headers
        if len(df.columns) == 3 and all(df.columns[i] in ['0', '1', '2'] for i in range(3)):
            df.columns = ["SMILES", "Name", "pIC50"]
        
        # Show data preview
        with st.expander("📊 Input Data Preview", expanded=True):
            st.dataframe(df.head(10))
            st.metric("Total Molecules", len(df))
        
        # Process data
        progress_bar = st.progress(0, text="Starting analysis...")
        df_processed = load_data(df)
        
        if df_processed is not None:
            # Generate MMPs
            row_df, delta_df, mmp_df = generate_mmps(df_processed, progress_bar)
            
            if mmp_df is not None:
                # Display results
                st.markdown('<h2 class="sub-header">📈 MMP Analysis Results</h2>', unsafe_allow_html=True)
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Transforms", len(mmp_df))
                with col2:
                    st.metric("Total MMP Pairs", len(delta_df))
                with col3:
                    avg_delta = mmp_df['mean_delta'].mean()
                    st.metric("Average ΔpIC50", f"{avg_delta:.2f}")
                with col4:
                    max_count = mmp_df['Count'].max()
                    st.metric("Max Transform Frequency", max_count)
                
                # Generate visualizations
                st.markdown('<h3 class="sub-header">🎨 Transform Visualizations</h3>', unsafe_allow_html=True)
                
                # Add images to mmp_df
                mmp_df['MMP Transform'] = mmp_df['rxn_mol'].apply(rxn_to_base64_image)
                mmp_df['Delta Distribution'] = mmp_df['Deltas'].apply(stripplot_base64_image)
                
                # Sort and display
                mmp_df.sort_values("mean_delta", inplace=True, ascending=sort_ascending)
                
                # Create display dataframe
                display_df = mmp_df[['MMP Transform', 'Count', 'mean_delta', 'Delta Distribution']].copy()
                display_df['mean_delta'] = display_df['mean_delta'].round(3)
                
                # Show results
                st.markdown(f"**Showing top {min(rows_to_show, len(display_df))} transforms:**")
                st.markdown(display_df.head(rows_to_show).to_html(escape=False), unsafe_allow_html=True)
                
                # Show raw data if requested
                if show_raw_data:
                    with st.expander("📋 Raw Fragment Data"):
                        st.dataframe(row_df.head(20))
                    
                    with st.expander("📋 Raw Delta Data"):
                        st.dataframe(delta_df.head(20))
                    
                    with st.expander("📋 Full MMP Data"):
                        st.dataframe(mmp_df)
                
                # Download buttons
                col1, col2, col3 = st.columns(3)
                with col1:
                    csv = mmp_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download MMP Results",
                        data=csv,
                        file_name="mmp_results.csv",
                        mime="text/csv"
                    )
                with col2:
                    csv = delta_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Delta Data",
                        data=csv,
                        file_name="delta_data.csv",
                        mime="text/csv"
                    )
                with col3:
                    csv = row_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Fragment Data",
                        data=csv,
                        file_name="fragment_data.csv",
                        mime="text/csv"
                    )
                
                # Visualization of delta distribution
                st.markdown('<h3 class="sub-header">📊 ΔpIC50 Distribution</h3>', unsafe_allow_html=True)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                all_deltas = []
                for deltas in mmp_df['Deltas']:
                    all_deltas.extend(deltas)
                
                sns.histplot(all_deltas, bins=30, kde=True, ax=ax)
                ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
                ax.set_xlabel('ΔpIC50')
                ax.set_ylabel('Frequency')
                ax.set_title('Distribution of All ΔpIC50 Values')
                st.pyplot(fig)
                
            else:
                st.error("Failed to generate MMPs. Please check your input data.")
        
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.exception(e)

elif not uploaded_file:
    st.info("👈 Please upload a CSV file to begin analysis.")
    
    # Sample data structure
    st.markdown("""
    ### Expected CSV Format:
    Your CSV file should have the following columns:
    
    1. **SMILES** - Molecular structures in SMILES format
    2. **Name** - Compound names or identifiers
    3. **pIC50** - Activity values (-logIC50)
    
    ### Example Data:
    ```
    SMILES,Name,pIC50
    Cc1ccccc1CO,Compound_1,6.5
    Cc1ccccc1CN,Compound_2,7.2
    Cc1ccc(OC)cc1,Compound_3,5.8
    ```
    
    ### What this app does:
    1. **Fragment Generation**: Cuts each molecule once to generate molecular pairs
    2. **MMP Identification**: Identifies matched molecular pairs with common cores
    3. **ΔpIC50 Calculation**: Calculates activity differences for each transform
    4. **Visualization**: Displays transforms and their ΔpIC50 distributions
    """)
    
    # Add example download
    example_data = """SMILES,Name,pIC50
Cc1ccccc1CO,Compound_1,6.5
Cc1ccccc1CN,Compound_2,7.2
Cc1ccc(OC)cc1,Compound_3,5.8
Cc1ccc(N)cc1,Compound_4,6.8
Cc1ccc(Cl)cc1,Compound_5,7.5
CCc1ccccc1,Compound_6,6.2
CCc1ccccc1O,Compound_7,6.9
CCc1ccccc1N,Compound_8,7.1"""
    
    st.download_button(
        label="📥 Download Example CSV",
        data=example_data,
        file_name="example_mmp_data.csv",
        mime="text/csv"
    )

# Requirements installation instructions
with st.expander("🔧 Installation Requirements"):
    st.markdown("""
    ### Required Python Packages:
    ```bash
    pip install streamlit pandas numpy rdkit-pypi useful_rdkit_utils matplotlib seaborn
    ```
    
    ### For Google Colab users:
    The original code includes Colab-specific installation cells. For Streamlit, 
    make sure to install all required packages in your environment.
    
    ### Running the App:
    ```bash
    streamlit run app.py
    ```
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>MMP Analysis Tool • Built with Streamlit and RDKit</p>
    </div>
    """,
    unsafe_allow_html=True
)
