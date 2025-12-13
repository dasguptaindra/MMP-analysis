# app.py
import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
import useful_rdkit_utils as uru
from scaffold_finder import FragmentMol
from operator import itemgetter
from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
import requests
from tqdm import tqdm
import sys

# Set page config
st.set_page_config(
    page_title="Matched Molecular Pairs Analyzer",
    page_icon="🧪",
    layout="wide"
)

# Title and description
st.title("🧪 Matched Molecular Pairs Analysis")
st.markdown("""
This application performs Matched Molecular Pair (MMP) analysis on hERG inhibition data from ChEMBL.
MMPs identify pairs of molecules that differ by a single structural transformation, allowing analysis
of how specific chemical changes affect biological activity (pIC50).
""")

# Download necessary files if needed
@st.cache_resource
def download_scaffold_finder():
    """Download scaffold_finder.py from GitHub"""
    try:
        response = requests.get("https://raw.githubusercontent.com/PatWalters/practical_cheminformatics_tutorials/main/sar_analysis/scaffold_finder.py")
        with open("scaffold_finder.py", "w") as f:
            f.write(response.text)
        return True
    except:
        st.error("Failed to download scaffold_finder.py")
        return False

# Download the file
if download_scaffold_finder():
    st.success("✓ Required libraries downloaded")

# Define utility functions
def remove_map_nums(mol):
    """Remove atom map numbers from a molecule"""
    for atm in mol.GetAtoms():
        atm.SetAtomMapNum(0)
    return mol

def sort_fragments(mol):
    """
    Transform a molecule with multiple fragments into a list of molecules 
    sorted by number of atoms from largest to smallest
    """
    frag_list = list(Chem.GetMolFrags(mol, asMols=True))
    frag_list = [remove_map_nums(x) for x in frag_list]
    frag_num_atoms_list = [(x.GetNumAtoms(), x) for x in frag_list]
    frag_num_atoms_list.sort(key=itemgetter(0), reverse=True)
    return [x[1] for x in frag_num_atoms_list]

def rxn_to_image(rxn):
    """Convert an RDKit reaction to an image"""
    drawer = rdMolDraw2D.MolDraw2DCairo(400, 200)
    drawer.DrawReaction(rxn)
    drawer.FinishDrawing()
    img_data = drawer.GetDrawingText()
    return img_data

def plot_distribution(dist, title="Delta Distribution"):
    """Plot a distribution as a seaborn stripplot"""
    fig, ax = plt.subplots(figsize=(8, 2))
    sns.stripplot(x=dist, ax=ax, size=8, alpha=0.7)
    ax.axvline(0, ls="--", c="red", alpha=0.7)
    ax.set_xlim(-5, 5)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("ΔpIC50")
    plt.tight_layout()
    return fig

# Sidebar for parameters
st.sidebar.header("Analysis Parameters")

# Load data button
if st.sidebar.button("📥 Load hERG Dataset", use_container_width=True):
    with st.spinner("Loading data from GitHub..."):
        try:
            # Read the data
            df = pd.read_csv("https://raw.githubusercontent.com/PatWalters/practical_cheminformatics_tutorials/main/data/hERG.csv")
            
            # Add RDKit molecule column
            df['mol'] = df.SMILES.apply(Chem.MolFromSmiles)
            
            # Remove salts, counterions, etc.
            df['mol'] = df['mol'].apply(uru.get_largest_fragment)
            
            # Store in session state
            st.session_state.df = df
            st.session_state.data_loaded = True
            
            st.sidebar.success(f"✓ Loaded {len(df)} molecules")
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")

# Parameters
if 'data_loaded' in st.session_state and st.session_state.data_loaded:
    min_transform_occurrence = st.sidebar.slider(
        "Minimum Transform Occurrence",
        min_value=2, max_value=20, value=5
    )
    
    rows_to_show = st.sidebar.slider(
        "Number of MMPs to Display",
        min_value=5, max_value=50, value=10
    )
    
    sort_ascending = st.sidebar.radio(
        "Sort Order",
        ["Increase Activity (negative Δ)", "Decrease Activity (positive Δ)"],
        index=0
    )
    
    ascending = True if sort_ascending.startswith("Increase") else False
    
    # Process data button
    if st.sidebar.button("🔬 Run MMP Analysis", use_container_width=True):
        with st.spinner("Decomposing molecules and finding MMPs..."):
            try:
                df = st.session_state.df
                
                # Progress bar
                progress_bar = st.progress(0)
                
                # Step 1: Decompose molecules to scaffolds and sidechains
                st.write("### Step 1: Decomposing molecules...")
                row_list = []
                total = len(df)
                
                for i, (smiles, name, pIC50, mol) in enumerate(df.values):
                    frag_list = FragmentMol(mol, maxCuts=1)
                    for _, frag_mol in frag_list:
                        pair_list = sort_fragments(frag_mol)
                        tmp_list = [smiles] + [Chem.MolToSmiles(x) for x in pair_list] + [name, pIC50]
                        row_list.append(tmp_list)
                    progress_bar.progress((i + 1) / total * 0.33)
                
                row_df = pd.DataFrame(row_list, columns=["SMILES", "Core", "R_group", "Name", "pIC50"])
                
                # Step 2: Collect pairs with the same scaffold
                st.write("### Step 2: Finding molecular pairs...")
                delta_list = []
                groups = list(row_df.groupby("Core"))
                
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
                    progress_bar.progress(0.33 + (idx + 1) / len(groups) * 0.33)
                
                cols = ["SMILES_1", "Core_1", "R_group_1", "Name_1", "pIC50_1",
                       "SMILES_2", "Core_2", "Rgroup_1", "Name_2", "pIC50_2",
                       "Transform", "Delta"]
                delta_df = pd.DataFrame(delta_list, columns=cols)
                
                # Step 3: Collect frequently occurring pairs
                st.write("### Step 3: Analyzing transforms...")
                mmp_list = []
                transform_groups = list(delta_df.groupby("Transform"))
                
                for idx, (k, v) in enumerate(transform_groups):
                    if len(v) > min_transform_occurrence:
                        mmp_list.append([k, len(v), v.Delta.values])
                    progress_bar.progress(0.66 + (idx + 1) / len(transform_groups) * 0.34)
                
                mmp_df = pd.DataFrame(mmp_list, columns=["Transform", "Count", "Deltas"])
                mmp_df['idx'] = range(0, len(mmp_df))
                mmp_df['mean_delta'] = [x.mean() for x in mmp_df.Deltas]
                mmp_df['rxn_mol'] = mmp_df.Transform.apply(lambda x: AllChem.ReactionFromSmarts(x, useSmiles=True))
                
                # Create index linking dataframes
                transform_dict = dict([(a, b) for a, b in mmp_df[["Transform", "idx"]].values])
                delta_df['idx'] = [transform_dict.get(x) for x in delta_df.Transform]
                
                # Store results in session state
                st.session_state.row_df = row_df
                st.session_state.delta_df = delta_df
                st.session_state.mmp_df = mmp_df
                st.session_state.analysis_complete = True
                
                progress_bar.progress(1.0)
                st.success(f"✓ Analysis complete! Found {len(mmp_df)} MMPs with occurrence ≥ {min_transform_occurrence}")
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")

# Main display area
if 'analysis_complete' in st.session_state and st.session_state.analysis_complete:
    st.header("📊 MMP Analysis Results")
    
    # Display statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Molecules", len(st.session_state.df))
    with col2:
        st.metric("Total Pairs", len(st.session_state.delta_df))
    with col3:
        st.metric("Unique MMPs", len(st.session_state.mmp_df))
    with col4:
        avg_delta = st.session_state.mmp_df['mean_delta'].mean()
        st.metric("Avg ΔpIC50", f"{avg_delta:.2f}")
    
    # Display MMP table
    st.subheader(f"Top {rows_to_show} MMP Transforms")
    
    # Sort the dataframe
    display_df = st.session_state.mmp_df.copy()
    display_df = display_df.sort_values("mean_delta", ascending=ascending)
    display_df = display_df.head(rows_to_show)
    
    # Create columns for display
    for idx, row in display_df.iterrows():
        with st.expander(f"Transform {row['idx']}: {row['Transform'][:50]}... (Count: {row['Count']}, Mean Δ: {row['mean_delta']:.2f})", expanded=False):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Display reaction image
                img_data = rxn_to_image(row['rxn_mol'])
                st.image(img_data, caption="Molecular Transformation", use_column_width=True)
                
                st.write(f"**Transform:** {row['Transform']}")
                st.write(f"**Occurrences:** {row['Count']}")
                st.write(f"**Mean ΔpIC50:** {row['mean_delta']:.2f}")
            
            with col2:
                # Display distribution plot
                fig = plot_distribution(row['Deltas'], f"ΔpIC50 Distribution (n={row['Count']})")
                st.pyplot(fig)
                
                # Show molecules with this transform
                st.write("**Example Pairs:**")
                pairs_df = st.session_state.delta_df[st.session_state.delta_df['idx'] == row['idx']].head(5)
                
                for _, pair in pairs_df.iterrows():
                    st.write(f"- **{pair['Name_1']}** (pIC50={pair['pIC50_1']:.2f}) → **{pair['Name_2']}** (pIC50={pair['pIC50_2']:.2f}) | Δ={pair['Delta']:.2f}")
    
    # Detailed data section
    st.subheader("📋 Detailed Data")
    
    tab1, tab2, tab3 = st.tabs(["MMP Summary", "All Pairs", "Raw Data"])
    
    with tab1:
        st.dataframe(st.session_state.mmp_df[['Transform', 'Count', 'mean_delta', 'idx']].sort_values("mean_delta", ascending=ascending), use_container_width=True)
    
    with tab2:
        st.dataframe(st.session_state.delta_df, use_container_width=True)
    
    with tab3:
        st.dataframe(st.session_state.df, use_container_width=True)
    
    # Download options
    st.subheader("📥 Download Results")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = st.session_state.mmp_df.to_csv(index=False)
        st.download_button(
            label="Download MMP Summary",
            data=csv,
            file_name="mmp_summary.csv",
            mime="text/csv"
        )
    
    with col2:
        csv = st.session_state.delta_df.to_csv(index=False)
        st.download_button(
            label="Download All Pairs",
            data=csv,
            file_name="all_pairs.csv",
            mime="text/csv"
        )
    
    with col3:
        csv = st.session_state.df.to_csv(index=False)
        st.download_button(
            label="Download Raw Data",
            data=csv,
            file_name="herg_data.csv",
            mime="text/csv"
        )

else:
    # Initial state
    st.info("👈 Click 'Load hERG Dataset' in the sidebar to begin the analysis.")
    
    # Show example of what the app does
    st.markdown("""
    ### How it works:
    1. **Load Data**: Fetches hERG inhibition data from ChEMBL
    2. **Decompose Molecules**: Breaks molecules into scaffold/R-group pairs
    3. **Find MMPs**: Identifies pairs of molecules that differ by a single transformation
    4. **Analyze Impact**: Calculates how each transformation affects pIC50
    
    ### Key Outputs:
    - **MMP Transforms**: Chemical transformations that occur multiple times
    - **ΔpIC50 Distribution**: How the transformation affects activity
    - **Statistical Summary**: Mean impact and occurrence count
    
    ### Example Insights:
    - Replace chlorobenzyl with benzyl → reduces hERG activity
    - Add methyl group at position X → increases activity by 0.5 pIC50 units
    """)

# Footer
st.markdown("---")
st.markdown("""
**References:**
- Hussain, J. & Rea, C. (2010). *Journal of Chemical Information and Modeling*, 50(3), 339-348.
- Dossetter, A. G. et al. (2013). *Drug Discovery Today*, 18(15-16), 724-731.
- Tyrchan, C. & Evertsson, E. (2017). *Computational and Structural Biotechnology Journal*, 15, 86-90.
""")
