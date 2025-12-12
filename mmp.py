# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
import io
import base64
from operator import itemgetter
from itertools import combinations
import tempfile
import os
import sys
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="MMP Analysis Tool",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2563EB;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #E5E7EB;
        padding-bottom: 0.5rem;
    }
    .transform-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border-left: 4px solid #3B82F6;
    }
    .metric-card {
        background-color: #F0F9FF;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🧪 Matched Molecular Pair (MMP) Analysis Tool</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 📋 Configuration")
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    # Parameters
    st.markdown("### ⚙️ Parameters")
    min_occurrence = st.slider("Minimum transform occurrences", 1, 20, 5, help="Minimum number of occurrences for a transform to be considered")
    
    # Display options
    st.markdown("### 👀 Display Options")
    show_all_transforms = st.checkbox("Show all transformations", value=False)
    transforms_to_display = st.slider("Number of transforms to display", 1, 50, 10, disabled=show_all_transforms)
    
    # Analysis options
    st.markdown("### 🔬 Analysis")
    show_top_positive = st.checkbox("Show top positive transforms", value=True)
    show_top_negative = st.checkbox("Show top negative transforms", value=True)
    show_compound_examples = st.checkbox("Show compound examples", value=True)
    
    # Save options
    st.markdown("### 💾 Export")
    save_results = st.checkbox("Save results to Excel")
    
    # About section
    with st.expander("About MMP Analysis"):
        st.markdown("""
        **Matched Molecular Pairs (MMP)** is a technique for identifying structural transformations that affect biological activity.
        
        **Key steps:**
        1. Decompose molecules into core and R-groups
        2. Find pairs with same core but different R-groups
        3. Calculate ΔpIC50 for each pair
        4. Identify frequently occurring transformations
        
        **References:**
        - Hussain & Rea (2010) *J. Chem. Inf. Model.*
        - Dossetter et al. (2013) *Drug Discovery Today*
        """)

# Helper functions
@st.cache_data
def load_data(file):
    """Load and preprocess data"""
    if file is not None:
        df = pd.read_csv(file)
        # Check required columns
        required_cols = ['SMILES', 'pIC50']
        if not all(col in df.columns for col in required_cols):
            st.error(f"CSV must contain columns: {required_cols}")
            return None
        
        # Convert SMILES to molecules
        df['mol'] = df['SMILES'].apply(Chem.MolFromSmiles)
        
        # Get largest fragment
        def get_largest_fragment(mol):
            if mol is None:
                return None
            frags = Chem.GetMolFrags(mol, asMols=True)
            if frags:
                return max(frags, key=lambda x: x.GetNumAtoms())
            return mol
        
        df['mol'] = df['mol'].apply(get_largest_fragment)
        return df
    return None

def remove_map_nums(mol):
    """Remove atom map numbers from a molecule"""
    if mol is None:
        return None
    for atm in mol.GetAtoms():
        atm.SetAtomMapNum(0)
    return mol

def sort_fragments(mol):
    """Sort fragments by number of atoms"""
    if mol is None:
        return []
    frag_list = list(Chem.GetMolFrags(mol, asMols=True))
    frag_list = [remove_map_nums(x) for x in frag_list]
    frag_num_atoms_list = [(x.GetNumAtoms(), x) for x in frag_list]
    frag_num_atoms_list.sort(key=itemgetter(0), reverse=True)
    return [x[1] for x in frag_num_atoms_list]

def FragmentMol(mol, maxCuts=1):
    """Simple fragmentation function"""
    # This is a simplified version - in practice you might want a more sophisticated fragmentation
    results = []
    try:
        # Try to break single bonds
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.SINGLE:
                # Create a copy and break the bond
                emol = Chem.EditableMol(mol)
                emol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                frag_mol = emol.GetMol()
                results.append((f"CUT_{bond.GetIdx()}", frag_mol))
    except:
        pass
    
    # Return at least the original molecule
    if not results:
        results.append(("NO_CUT", mol))
    
    return results

def perform_mmp_analysis(df, min_transform_occurrence):
    """Perform MMP analysis"""
    if df is None or len(df) == 0:
        return None, None
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 1: Decompose molecules
    status_text.text("Step 1/4: Decomposing molecules...")
    progress_bar.progress(25)
    
    row_list = []
    for idx, row in df.iterrows():
        smiles = row['SMILES']
        name = row.get('Name', f"CMPD_{idx}")
        pIC50 = row['pIC50']
        mol = row['mol']
        
        if mol is None:
            continue
            
        frag_list = FragmentMol(mol, maxCuts=1)
        for _, frag_mol in frag_list:
            pair_list = sort_fragments(frag_mol)
            if len(pair_list) >= 2:
                tmp_list = [smiles] + [Chem.MolToSmiles(x) for x in pair_list] + [name, pIC50]
                row_list.append(tmp_list)
    
    if not row_list:
        st.error("No valid fragments found")
        return None, None
    
    row_df = pd.DataFrame(row_list, columns=["SMILES", "Core", "R_group", "Name", "pIC50"])
    
    # Step 2: Collect pairs
    status_text.text("Step 2/4: Collecting molecular pairs...")
    progress_bar.progress(50)
    
    delta_list = []
    for k, v in row_df.groupby("Core"):
        if len(v) > 1:
            for a, b in combinations(range(0, len(v)), 2):
                reagent_a = v.iloc[a]
                reagent_b = v.iloc[b]
                if reagent_a.SMILES == reagent_b.SMILES:
                    continue
                
                # Sort by SMILES for canonical ordering
                reagent_a, reagent_b = sorted([reagent_a, reagent_b], key=lambda x: x.SMILES)
                delta = reagent_b.pIC50 - reagent_a.pIC50
                transform_str = f"{reagent_a.R_group.replace('*', '*-')}>>{reagent_b.R_group.replace('*', '*-')}"
                
                delta_list.append([
                    reagent_a.SMILES, reagent_a.Core, reagent_a.R_group, reagent_a.Name, reagent_a.pIC50,
                    reagent_b.SMILES, reagent_b.Core, reagent_b.R_group, reagent_b.Name, reagent_b.pIC50,
                    transform_str, delta
                ])
    
    if not delta_list:
        st.error("No molecular pairs found")
        return None, None
    
    cols = [
        "SMILES_1", "Core_1", "R_group_1", "Name_1", "pIC50_1",
        "SMILES_2", "Core_2", "Rgroup_2", "Name_2", "pIC50_2",
        "Transform", "Delta"
    ]
    delta_df = pd.DataFrame(delta_list, columns=cols)
    
    # Step 3: Collect frequent transforms
    status_text.text("Step 3/4: Analyzing transformations...")
    progress_bar.progress(75)
    
    mmp_list = []
    for k, v in delta_df.groupby("Transform"):
        if len(v) >= min_transform_occurrence:
            mmp_list.append([k, len(v), v.Delta.values])
    
    if not mmp_list:
        st.warning(f"No transforms found with {min_transform_occurrence}+ occurrences")
        return delta_df, None
    
    mmp_df = pd.DataFrame(mmp_list, columns=["Transform", "Count", "Deltas"])
    mmp_df['idx'] = range(0, len(mmp_df))
    mmp_df['mean_delta'] = [x.mean() for x in mmp_df.Deltas]
    mmp_df['rxn_mol'] = mmp_df.Transform.apply(
        lambda x: AllChem.ReactionFromSmarts(x.replace('*-', '*'), useSmiles=True) if pd.notna(x) else None
    )
    
    # Step 4: Complete
    status_text.text("Step 4/4: Analysis complete!")
    progress_bar.progress(100)
    status_text.empty()
    progress_bar.empty()
    
    return delta_df, mmp_df

def plot_stripplot_to_fig(deltas):
    """Create a stripplot figure"""
    fig, ax = plt.subplots(figsize=(4, 1.5))
    sns.stripplot(x=deltas, ax=ax, jitter=0.2, alpha=0.7, s=5, color='blue')
    ax.axvline(0, ls='--', c='red')
    ax.set_xlim(-5, 5)
    ax.set_xlabel('ΔpIC50')
    ax.set_yticks([])
    plt.tight_layout()
    return fig

def get_rxn_image(rxn_mol):
    """Convert reaction to base64 image"""
    if rxn_mol is None:
        return None
    img = Draw.ReactionToImage(rxn_mol, subImgSize=(300, 150))
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def find_examples(delta_df, transform):
    """Find examples for a specific transform"""
    examples = delta_df[delta_df['Transform'] == transform]
    if len(examples) == 0:
        return None
    
    example_list = []
    for _, row in examples.sort_values("Delta", ascending=False).iterrows():
        # Create two entries for each pair
        example_list.append({
            "SMILES": row['SMILES_1'],
            "Name": row['Name_1'],
            "pIC50": row['pIC50_1'],
            "Type": "Before"
        })
        example_list.append({
            "SMILES": row['SMILES_2'],
            "Name": row['Name_2'],
            "pIC50": row['pIC50_2'],
            "Type": "After"
        })
    
    return pd.DataFrame(example_list)

# Main app logic
if uploaded_file is not None:
    # Load data
    df = load_data(uploaded_file)
    
    if df is not None:
        # Show dataset info
        with st.expander("📊 Dataset Overview", expanded=True):
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Compounds", len(df))
            col2.metric("Min pIC50", f"{df['pIC50'].min():.2f}")
            col3.metric("Max pIC50", f"{df['pIC50'].max():.2f}")
            
            st.dataframe(df[['SMILES', 'Name', 'pIC50']].head(10), use_container_width=True)
        
        # Perform MMP analysis
        st.markdown('<h2 class="section-header">🔍 MMP Analysis Results</h2>', unsafe_allow_html=True)
        
        delta_df, mmp_df = perform_mmp_analysis(df, min_occurrence)
        
        if delta_df is not None:
            # Show statistics
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Pairs", len(delta_df))
            
            if mmp_df is not None:
                col2.metric("Unique Transforms", len(mmp_df))
                col3.metric("Avg Transform Frequency", f"{mmp_df['Count'].mean():.1f}")
                
                # Sort transforms
                mmp_df_sorted = mmp_df.sort_values("mean_delta", ascending=False)
                
                # Show top positive transforms
                if show_top_positive:
                    st.markdown('<h3 class="section-header">📈 Top Positive Transformations</h3>', unsafe_allow_html=True)
                    top_positive = mmp_df_sorted.head(3)
                    
                    for i, (_, row) in enumerate(top_positive.iterrows()):
                        with st.container():
                            col1, col2, col3 = st.columns([1, 2, 1])
                            
                            with col1:
                                img_base64 = get_rxn_image(row['rxn_mol'])
                                if img_base64:
                                    st.markdown(f'<img src="data:image/png;base64,{img_base64}" width="300">', unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown(f"""
                                <div class="transform-card">
                                    <h4>Transform #{i+1}</h4>
                                    <p><strong>Transformation:</strong> {row['Transform']}</p>
                                    <p><strong>Mean ΔpIC50:</strong> {row['mean_delta']:.2f}</p>
                                    <p><strong>Occurrences:</strong> {row['Count']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col3:
                                fig = plot_stripplot_to_fig(row['Deltas'])
                                st.pyplot(fig)
                            
                            # Show compound examples
                            if show_compound_examples:
                                examples_df = find_examples(delta_df, row['Transform'])
                                if examples_df is not None and len(examples_df) > 0:
                                    with st.expander(f"View {len(examples_df)//2} compound pairs for this transform"):
                                        # Display molecules in a grid
                                        cols = st.columns(4)
                                        for idx, (_, example_row) in enumerate(examples_df.iterrows()):
                                            mol = Chem.MolFromSmiles(example_row['SMILES'])
                                            if mol:
                                                with cols[idx % 4]:
                                                    img = Draw.MolToImage(mol, size=(200, 200))
                                                    st.image(img, caption=f"{example_row['Name']} (pIC50: {example_row['pIC50']:.2f})")
                                        
                                        # Also show as table
                                        st.dataframe(examples_df, use_container_width=True)
                
                # Show top negative transforms
                if show_top_negative:
                    st.markdown('<h3 class="section-header">📉 Top Negative Transformations</h3>', unsafe_allow_html=True)
                    top_negative = mmp_df_sorted.tail(3).iloc[::-1]
                    
                    for i, (_, row) in enumerate(top_negative.iterrows()):
                        with st.container():
                            col1, col2, col3 = st.columns([1, 2, 1])
                            
                            with col1:
                                img_base64 = get_rxn_image(row['rxn_mol'])
                                if img_base64:
                                    st.markdown(f'<img src="data:image/png;base64,{img_base64}" width="300">', unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown(f"""
                                <div class="transform-card">
                                    <h4>Transform #{i+1} (Negative)</h4>
                                    <p><strong>Transformation:</strong> {row['Transform']}</p>
                                    <p><strong>Mean ΔpIC50:</strong> {row['mean_delta']:.2f}</p>
                                    <p><strong>Occurrences:</strong> {row['Count']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col3:
                                fig = plot_stripplot_to_fig(row['Deltas'])
                                st.pyplot(fig)
                            
                            # Show compound examples
                            if show_compound_examples:
                                examples_df = find_examples(delta_df, row['Transform'])
                                if examples_df is not None and len(examples_df) > 0:
                                    with st.expander(f"View {len(examples_df)//2} compound pairs for this transform"):
                                        # Display molecules in a grid
                                        cols = st.columns(4)
                                        for idx, (_, example_row) in enumerate(examples_df.iterrows()):
                                            mol = Chem.MolFromSmiles(example_row['SMILES'])
                                            if mol:
                                                with cols[idx % 4]:
                                                    img = Draw.MolToImage(mol, size=(200, 200))
                                                    st.image(img, caption=f"{example_row['Name']} (pIC50: {example_row['pIC50']:.2f})")
                                        
                                        # Also show as table
                                        st.dataframe(examples_df, use_container_width=True)
                
                # Show all transforms table
                st.markdown('<h3 class="section-header">📋 All Transformations</h3>', unsafe_allow_html=True)
                
                if show_all_transforms:
                    display_df = mmp_df_sorted
                else:
                    display_df = mmp_df_sorted.head(transforms_to_display)
                
                # Prepare display dataframe
                display_data = []
                for _, row in display_df.iterrows():
                    # Create stripplot
                    fig = plot_stripplot_to_fig(row['Deltas'])
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', bbox_inches='tight')
                    plt.close(fig)
                    stripplot_b64 = base64.b64encode(buf.getvalue()).decode()
                    
                    # Create reaction image
                    rxn_img_b64 = get_rxn_image(row['rxn_mol'])
                    
                    display_data.append({
                        "Transform": row['Transform'],
                        "Reaction": f'<img src="data:image/png;base64,{rxn_img_b64}" width="200">',
                        "Count": row['Count'],
                        "Mean ΔpIC50": f"{row['mean_delta']:.2f}",
                        "ΔpIC50 Distribution": f'<img src="data:image/png;base64,{stripplot_b64}" width="150">'
                    })
                
                # Convert to HTML table
                if display_data:
                    display_df_html = pd.DataFrame(display_data)
                    st.markdown(display_df_html.to_html(escape=False, index=False), unsafe_allow_html=True)
                
                # Export results
                if save_results and mmp_df is not None:
                    st.markdown('<h3 class="section-header">💾 Export Results</h3>', unsafe_allow_html=True)
                    
                    # Create downloadable files
                    @st.cache_data
                    def convert_df_to_csv(df):
                        return df.to_csv(index=False).encode('utf-8')
                    
                    @st.cache_data
                    def convert_df_to_excel(df):
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            df.to_excel(writer, index=False, sheet_name='MMP_Results')
                        return output.getvalue()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.download_button(
                            label="📥 Download MMP Results (CSV)",
                            data=convert_df_to_csv(mmp_df_sorted),
                            file_name="mmp_results.csv",
                            mime="text/csv"
                        )
                    
                    with col2:
                        st.download_button(
                            label="📥 Download MMP Results (Excel)",
                            data=convert_df_to_excel(mmp_df_sorted),
                            file_name="mmp_results.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    
                    # Also provide delta pairs
                    if st.checkbox("Include molecular pairs data"):
                        col3, col4 = st.columns(2)
                        with col3:
                            st.download_button(
                                label="📥 Download Pairs Data (CSV)",
                                data=convert_df_to_csv(delta_df),
                                file_name="mmp_pairs.csv",
                                mime="text/csv"
                            )
                        with col4:
                            st.download_button(
                                label="📥 Download Pairs Data (Excel)",
                                data=convert_df_to_excel(delta_df),
                                file_name="mmp_pairs.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
            
            else:
                st.info(f"No transformations found with {min_occurrence}+ occurrences. Try reducing the minimum occurrence threshold.")
else:
    # Show welcome message when no file is uploaded
    st.markdown("""
    ## Welcome to the MMP Analysis Tool! 👋
    
    This tool performs **Matched Molecular Pair (MMP) analysis** to identify structural transformations that affect compound potency.
    
    ### How to use:
    1. **Upload your data** using the sidebar on the left
    2. **Configure parameters** like minimum transform occurrences
    3. **View results** including top positive/negative transformations
    4. **Export findings** for further analysis
    
    ### Required CSV format:
    Your CSV file should contain at least these columns:
    - `SMILES`: Molecular structures in SMILES format
    - `pIC50`: Potency values (negative log of IC50)
    - `Name`: Compound names (optional but recommended)
    
    ### Example CSV format:
    ```csv
    SMILES,Name,pIC50
    CC(=O)OC1=CC=CC=C1C(=O)O,aspirin,5.0
    CN1C=NC2=C1C(=O)N(C(=O)N2C)C,caffeine,4.2
    ...
    ```
    
    ### References:
    - Hussain, J. & Rea, C. (2010). Computationally efficient algorithm to identify matched molecular pairs (MMPs) in large data sets. *Journal of Chemical Information and Modeling*, 50(3), 339-348.
    - Dossetter, A. G., Griffen, E. J., & Leach, A. G. (2013). Matched molecular pair analysis in drug discovery. *Drug Discovery Today*, 18(15-16), 724-731.
    
    ⬅️ **Upload a CSV file in the sidebar to get started!**
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; font-size: 0.9rem;">
    <p>MMP Analysis Tool v1.0 | Built with Streamlit, RDKit, and Pandas</p>
    <p>For research use only. Always validate computational predictions with experimental data.</p>
</div>
""", unsafe_allow_html=True)
