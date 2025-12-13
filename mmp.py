import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdMMPA import FragmentMol
from rdkit.Chem import AllChem, Draw, rdMolDraw2D
from rdkit.Chem.Draw import rdMolDraw2D
from operator import itemgetter
from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
from io import BytesIO
import tempfile
import sys
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="MMP Analysis Tool",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .stDataFrame {
        font-size: 0.9rem;
    }
    .css-1d391kg {
        padding-top: 2rem;
    }
    .reaction-img {
        display: block;
        margin: 0 auto;
        max-width: 100%;
    }
    .molecule-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🧪 Matched Molecular Pair (MMP) Analysis</h1>', unsafe_allow_html=True)
st.markdown("""
This tool performs Matched Molecular Pair analysis to identify structural transformations 
that consistently affect biological activity. Upload your data to get started.
""")

# Sidebar for file upload and parameters
with st.sidebar:
    st.markdown("## 📁 Data Upload")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file with SMILES and activity data",
        type=['csv'],
        help="CSV should contain columns: SMILES, Name, pIC50 (or similar activity column)"
    )
    
    st.markdown("## ⚙️ Analysis Parameters")
    
    min_transform_occurrence = st.slider(
        "Minimum Transform Occurrence",
        min_value=2,
        max_value=20,
        value=5,
        help="Minimum number of times a transformation must appear to be included"
    )
    
    show_n_transforms = st.slider(
        "Number of Transforms to Display",
        min_value=5,
        max_value=50,
        value=15
    )
    
    sort_by = st.selectbox(
        "Sort Transforms By",
        options=["Mean ΔpIC50 (Descending)", "Mean ΔpIC50 (Ascending)", "Count (Descending)"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 📊 Sample Data Format")
    st.code("""SMILES,Name,pIC50
CN(C)CCc1c[nH]c2ccc(OC)cc12,Compound_1,6.5
COc1ccc2c(c1)CCN(CCc1ccc(OC)cc1)C2,Compound_2,7.2
...""")
    
    st.markdown("---")
    st.markdown("### ℹ️ About MMP Analysis")
    st.info("""
    MMP analysis identifies pairs of molecules that differ by a single 
    structural transformation and correlates these changes with changes 
    in biological activity.
    """)

# Helper functions
def remove_map_nums(mol):
    """Remove atom map numbers from a molecule"""
    if mol is None:
        return mol
    for atm in mol.GetAtoms():
        atm.SetAtomMapNum(0)
    return mol

def sort_fragments(mol):
    """Transform a molecule with multiple fragments into a list of molecules sorted by number of atoms"""
    if mol is None:
        return []
    frag_list = list(Chem.GetMolFrags(mol, asMols=True))
    frag_list = [remove_map_nums(x) for x in frag_list]
    frag_num_atoms_list = [(x.GetNumAtoms(), x) for x in frag_list]
    frag_num_atoms_list.sort(key=itemgetter(0), reverse=True)
    return [x[1] for x in frag_num_atoms_list]

def get_largest_fragment(mol):
    """Get the largest fragment from a molecule"""
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True)
    if not frags:
        return mol
    return max(frags, key=lambda x: x.GetNumAtoms())

def rxn_to_base64_image(rxn):
    """Convert an RDKit reaction to a base64 encoded image"""
    try:
        if rxn is None:
            return ""
        drawer = rdMolDraw2D.MolDraw2DCairo(400, 200)
        drawer.DrawReaction(rxn)
        drawer.FinishDrawing()
        img_data = drawer.GetDrawingText()
        return base64.b64encode(img_data).decode('utf-8')
    except:
        return ""

def stripplot_base64_image(data):
    """Create a strip plot and return as base64 image"""
    try:
        fig, ax = plt.subplots(figsize=(4, 1.5))
        sns.stripplot(x=data, ax=ax, size=6, alpha=0.7)
        ax.axvline(0, ls="--", c="red", alpha=0.5)
        ax.set_xlim(-5, 5)
        ax.set_xlabel('ΔpIC50')
        ax.grid(True, alpha=0.3)
        
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
    except:
        return ""

def mol_to_smiles(mol):
    """Convert RDKit molecule to SMILES, handling None"""
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol)

# Main analysis function
def perform_mmp_analysis(df, min_occurrence=5):
    """Perform the complete MMP analysis pipeline"""
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 1: Prepare molecules
    status_text.text("Step 1/5: Preparing molecules...")
    if 'mol' not in df.columns:
        df['mol'] = df['SMILES'].apply(Chem.MolFromSmiles)
    
    # Remove invalid molecules
    initial_count = len(df)
    df = df[df['mol'].notna()].copy()
    valid_count = len(df)
    
    if valid_count == 0:
        st.error("No valid molecules found in the dataset!")
        return None, None, None
    
    st.info(f"Loaded {valid_count} valid molecules out of {initial_count} total.")
    
    # Get largest fragment for each molecule
    df['mol'] = df['mol'].apply(get_largest_fragment)
    progress_bar.progress(20)
    
    # Step 2: Generate fragments
    status_text.text("Step 2/5: Generating molecular fragments...")
    row_list = []
    
    for i, (smiles, name, pIC50, mol) in enumerate(df[['SMILES', 'Name', 'pIC50', 'mol']].values):
        try:
            frag_list = FragmentMol(mol, maxCuts=1)
            for _, frag_mol in frag_list:
                pair_list = sort_fragments(frag_mol)
                if len(pair_list) >= 2:
                    tmp_list = [smiles] + [mol_to_smiles(x) for x in pair_list] + [name, pIC50]
                    row_list.append(tmp_list)
        except:
            continue
    
    if not row_list:
        st.error("No fragments could be generated!")
        return None, None, None
    
    row_df = pd.DataFrame(row_list, columns=["SMILES", "Core", "R_group", "Name", "pIC50"])
    progress_bar.progress(40)
    
    # Step 3: Generate delta pairs
    status_text.text("Step 3/5: Generating matched pairs...")
    delta_list = []
    
    # Group by core and find pairs
    core_groups = list(row_df.groupby("Core"))
    total_groups = len(core_groups)
    
    for idx, (k, v) in enumerate(core_groups):
        if len(v) > 2:
            for a, b in combinations(range(len(v)), 2):
                reagent_a = v.iloc[a]
                reagent_b = v.iloc[b]
                
                if reagent_a.SMILES == reagent_b.SMILES:
                    continue
                
                # Sort by SMILES for consistency
                if reagent_a.SMILES > reagent_b.SMILES:
                    reagent_a, reagent_b = reagent_b, reagent_a
                
                delta = reagent_b.pIC50 - reagent_a.pIC50
                transform_str = f"{reagent_a.R_group.replace('*', '*-')}>>{reagent_b.R_group.replace('*', '*-')}"
                
                delta_list.append(
                    list(reagent_a.values) + 
                    list(reagent_b.values) + 
                    [transform_str, delta]
                )
    
    if not delta_list:
        st.error("No valid matched pairs found!")
        return None, None, None
    
    cols = [
        "SMILES_1", "Core_1", "R_group_1", "Name_1", "pIC50_1",
        "SMILES_2", "Core_2", "R_group_2", "Name_2", "pIC50_2",
        "Transform", "Delta"
    ]
    delta_df = pd.DataFrame(delta_list, columns=cols)
    progress_bar.progress(60)
    
    # Step 4: Aggregate transforms
    status_text.text("Step 4/5: Aggregating transformations...")
    mmp_list = []
    
    for k, v in delta_df.groupby("Transform"):
        if len(v) >= min_occurrence:
            mmp_list.append([k, len(v), v.Delta.values])
    
    if not mmp_list:
        st.warning(f"No transformations found with occurrence ≥ {min_occurrence}. Try lowering the threshold.")
        return None, None, None
    
    mmp_df = pd.DataFrame(mmp_list, columns=["Transform", "Count", "Deltas"])
    mmp_df['idx'] = range(len(mmp_df))
    mmp_df['mean_delta'] = [x.mean() for x in mmp_df.Deltas]
    mmp_df['std_delta'] = [x.std() for x in mmp_df.Deltas]
    
    # Create reaction molecules
    mmp_df['rxn_mol'] = mmp_df.Transform.apply(
        lambda x: AllChem.ReactionFromSmarts(x.replace('*-', '*'), useSmiles=True) 
        if x is not None else None
    )
    
    # Create transform dictionary
    transform_dict = dict(mmp_df[["Transform", "idx"]].values)
    delta_df['idx'] = delta_df.Transform.map(transform_dict)
    progress_bar.progress(80)
    
    # Step 5: Generate visualizations
    status_text.text("Step 5/5: Generating visualizations...")
    mmp_df['MMP Transform'] = mmp_df.rxn_mol.apply(rxn_to_base64_image)
    mmp_df['Delta Distribution'] = mmp_df.Deltas.apply(stripplot_base64_image)
    progress_bar.progress(100)
    
    status_text.text("Analysis complete!")
    
    return df, delta_df, mmp_df

def display_molecule_grid(molecules, title, n_cols=4):
    """Display a grid of molecules"""
    if not molecules:
        return
    
    st.markdown(f'<h3 class="sub-header">{title}</h3>', unsafe_allow_html=True)
    
    cols = st.columns(n_cols)
    for idx, (mol, name, pIC50) in enumerate(molecules):
        col_idx = idx % n_cols
        with cols[col_idx]:
            try:
                img = Draw.MolToImage(mol, size=(200, 150))
                st.image(img, caption=f"{name}\npIC50: {pIC50:.2f}")
            except:
                st.write(f"{name}: Invalid molecule")

# Main app logic
def main():
    # Example data
    example_data = pd.DataFrame({
        'SMILES': [
            'CN(C)CCc1c[nH]c2ccc(OC)cc12',
            'COc1ccc2c(c1)CCN(CCc1ccc(OC)cc1)C2',
            'COc1ccc2[nH]c(CCN(C)C)cc2c1',
            'CN(C)CCc1c[nH]c2ccc(O)cc12',
            'COc1ccc2c(c1)CCN(CCc1ccc(O)cc1)C2'
        ],
        'Name': ['Compound_1', 'Compound_2', 'Compound_3', 'Compound_4', 'Compound_5'],
        'pIC50': [6.5, 7.2, 6.8, 5.9, 6.1]
    })
    
    # Use uploaded file or example data
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Check required columns
            required_cols = ['SMILES', 'Name', 'pIC50']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
                st.info("Using example data instead.")
                df = example_data
            else:
                st.success(f"Successfully loaded {len(df)} molecules from {uploaded_file.name}")
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.info("Using example data instead.")
            df = example_data
    else:
        df = example_data
        st.info("Using example data. Upload your own CSV file to analyze your compounds.")
    
    # Display input data
    with st.expander("📋 View Input Data", expanded=False):
        st.dataframe(df, use_container_width=True)
        
        # Show molecule preview
        if 'mol' not in df.columns:
            df['mol'] = df['SMILES'].apply(Chem.MolFromSmiles)
        
        valid_mols = df[df['mol'].notna()]
        if len(valid_mols) > 0:
            st.markdown("### Molecule Preview")
            mols_to_show = min(8, len(valid_mols))
            mols = valid_mols['mol'].head(mols_to_show).tolist()
            names = valid_mols['Name'].head(mols_to_show).tolist()
            
            try:
                img = Draw.MolsToGridImage(
                    mols, 
                    molsPerRow=4, 
                    subImgSize=(200, 150),
                    legends=[f"{name}" for name in names]
                )
                st.image(img)
            except:
                st.write("Could not generate molecule preview")
    
    # Perform analysis when button is clicked
    if st.button("🚀 Perform MMP Analysis", type="primary", use_container_width=True):
        with st.spinner("Performing MMP analysis..."):
            df_processed, delta_df, mmp_df = perform_mmp_analysis(df.copy(), min_transform_occurrence)
        
        if mmp_df is not None:
            # Display results
            st.markdown("---")
            st.markdown('<h2 class="main-header">📊 MMP Analysis Results</h2>', unsafe_allow_html=True)
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Transforms", len(mmp_df))
            with col2:
                avg_effect = mmp_df['mean_delta'].abs().mean()
                st.metric("Avg |ΔpIC50|", f"{avg_effect:.2f}")
            with col3:
                max_effect = mmp_df['mean_delta'].abs().max()
                st.metric("Max |ΔpIC50|", f"{max_effect:.2f}")
            
            # Sort transforms based on selection
            if sort_by == "Mean ΔpIC50 (Descending)":
                mmp_df_sorted = mmp_df.sort_values("mean_delta", ascending=False)
            elif sort_by == "Mean ΔpIC50 (Ascending)":
                mmp_df_sorted = mmp_df.sort_values("mean_delta", ascending=True)
            else:  # Count (Descending)
                mmp_df_sorted = mmp_df.sort_values("Count", ascending=False)
            
            # Display transforms in tabs
            tab1, tab2, tab3 = st.tabs(["Transform Table", "Detailed View", "Export Data"])
            
            with tab1:
                st.markdown(f"### Top {show_n_transforms} Transforms")
                
                # Create display dataframe
                display_df = mmp_df_sorted.head(show_n_transforms).copy()
                
                # Create HTML table with images
                html_content = """
                <table style="width:100%; border-collapse: collapse;">
                    <thead>
                        <tr style="background-color: #3B82F6; color: white;">
                            <th style="padding: 10px; text-align: center;">Transform</th>
                            <th style="padding: 10px; text-align: center;">Count</th>
                            <th style="padding: 10px; text-align: center;">Mean ΔpIC50</th>
                            <th style="padding: 10px; text-align: center;">Std Dev</th>
                            <th style="padding: 10px; text-align: center;">Distribution</th>
                        </tr>
                    </thead>
                    <tbody>
                """
                
                for _, row in display_df.iterrows():
                    if row['MMP Transform']:
                        img_tag = f'<img src="data:image/png;base64,{row["MMP Transform"]}" style="max-width: 300px;">'
                    else:
                        img_tag = "N/A"
                    
                    if row['Delta Distribution']:
                        dist_tag = f'<img src="data:image/png;base64,{row["Delta Distribution"]}">'
                    else:
                        dist_tag = "N/A"
                    
                    html_content += f"""
                    <tr style="border-bottom: 1px solid #ddd;">
                        <td style="padding: 10px; text-align: center;">{img_tag}</td>
                        <td style="padding: 10px; text-align: center; font-weight: bold;">{row['Count']}</td>
                        <td style="padding: 10px; text-align: center; font-weight: bold; color: {'red' if row['mean_delta'] < 0 else 'green'}">
                            {row['mean_delta']:.2f}
                        </td>
                        <td style="padding: 10px; text-align: center;">{row['std_delta']:.2f}</td>
                        <td style="padding: 10px; text-align: center;">{dist_tag}</td>
                    </tr>
                    """
                
                html_content += "</tbody></table>"
                st.markdown(html_content, unsafe_allow_html=True)
            
            with tab2:
                st.markdown("### Detailed Transform View")
                
                if len(mmp_df_sorted) > 0:
                    # Select a transform to examine
                    transform_options = [f"{idx}: {row.Transform} (n={row.Count}, Δ={row.mean_delta:.2f})" 
                                        for idx, row in mmp_df_sorted.iterrows()]
                    
                    selected_transform = st.selectbox(
                        "Select a transform to examine:",
                        options=range(len(mmp_df_sorted)),
                        format_func=lambda x: transform_options[x]
                    )
                    
                    if selected_transform is not None:
                        row = mmp_df_sorted.iloc[selected_transform]
                        
                        # Display transform info
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.markdown("#### Transform Information")
                            st.metric("Mean ΔpIC50", f"{row['mean_delta']:.2f}")
                            st.metric("Number of Examples", row['Count'])
                            st.metric("Standard Deviation", f"{row['std_delta']:.2f}")
                            
                            # Show reaction image
                            if row['rxn_mol'] is not None:
                                try:
                                    img = Draw.ReactionToImage(row['rxn_mol'])
                                    st.image(img, caption="Reaction Diagram")
                                except:
                                    st.write("Could not generate reaction image")
                        
                        with col2:
                            st.markdown("#### Activity Distribution")
                            fig, ax = plt.subplots(figsize=(8, 4))
                            ax.hist(row['Deltas'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
                            ax.axvline(row['mean_delta'], color='red', linestyle='--', label=f'Mean: {row["mean_delta"]:.2f}')
                            ax.axvline(0, color='black', linestyle='-', alpha=0.3)
                            ax.set_xlabel('ΔpIC50')
                            ax.set_ylabel('Frequency')
                            ax.set_title('Distribution of Activity Changes')
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
                        
                        # Show example pairs
                        st.markdown("#### Example Molecular Pairs")
                        if delta_df is not None:
                            examples = delta_df[delta_df['idx'] == row['idx']].head(5)
                            
                            for _, example in examples.iterrows():
                                col1, col2, col3 = st.columns([2, 1, 2])
                                
                                with col1:
                                    try:
                                        mol1 = Chem.MolFromSmiles(example['SMILES_1'])
                                        img1 = Draw.MolToImage(mol1, size=(250, 200))
                                        st.image(img1, caption=f"{example['Name_1']}\npIC50: {example['pIC50_1']:.2f}")
                                    except:
                                        st.write(f"Molecule 1: {example['Name_1']}")
                                
                                with col2:
                                    delta_val = example['Delta']
                                    color = "red" if delta_val < 0 else "green"
                                    arrow = "↓" if delta_val < 0 else "↑"
                                    st.markdown(f"<h2 style='text-align: center; color: {color};'>{arrow} {delta_val:.2f}</h2>", 
                                                unsafe_allow_html=True)
                                    st.markdown(f"<p style='text-align: center;'>ΔpIC50</p>", unsafe_allow_html=True)
                                
                                with col3:
                                    try:
                                        mol2 = Chem.MolFromSmiles(example['SMILES_2'])
                                        img2 = Draw.MolToImage(mol2, size=(250, 200))
                                        st.image(img2, caption=f"{example['Name_2']}\npIC50: {example['pIC50_2']:.2f}")
                                    except:
                                        st.write(f"Molecule 2: {example['Name_2']}")
                                
                                st.markdown("---")
            
            with tab3:
                st.markdown("### Export Results")
                
                # Create downloadable DataFrames
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**MMP Transforms**")
                    st.dataframe(mmp_df_sorted[['Transform', 'Count', 'mean_delta', 'std_delta']], use_container_width=True)
                    
                    # Download button for transforms
                    csv1 = mmp_df_sorted[['Transform', 'Count', 'mean_delta', 'std_delta']].to_csv(index=False)
                    st.download_button(
                        label="Download Transforms CSV",
                        data=csv1,
                        file_name="mmp_transforms.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    st.markdown("**Molecular Pairs**")
                    if delta_df is not None:
                        display_cols = ['SMILES_1', 'Name_1', 'pIC50_1', 'SMILES_2', 'Name_2', 'pIC50_2', 'Transform', 'Delta']
                        available_cols = [col for col in display_cols if col in delta_df.columns]
                        st.dataframe(delta_df[available_cols].head(20), use_container_width=True)
                        
                        # Download button for pairs
                        csv2 = delta_df[available_cols].to_csv(index=False)
                        st.download_button(
                            label="Download Pairs CSV",
                            data=csv2,
                            file_name="mmp_pairs.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                
                # Export all data
                st.markdown("---")
                st.markdown("**Export Complete Analysis**")
                
                # Create a zip file with all data
                import zipfile
                from io import BytesIO
                
                if st.button("📦 Create Complete Export Package", use_container_width=True):
                    buffer = BytesIO()
                    with zipfile.ZipFile(buffer, 'w') as zip_file:
                        # Add transforms
                        transforms_csv = mmp_df_sorted.to_csv(index=False)
                        zip_file.writestr('mmp_transforms.csv', transforms_csv)
                        
                        # Add pairs
                        if delta_df is not None:
                            pairs_csv = delta_df.to_csv(index=False)
                            zip_file.writestr('mmp_pairs.csv', pairs_csv)
                        
                        # Add input data
                        input_csv = df.to_csv(index=False)
                        zip_file.writestr('input_data.csv', input_csv)
                    
                    buffer.seek(0)
                    
                    st.download_button(
                        label="Download ZIP Archive",
                        data=buffer,
                        file_name="mmp_analysis_results.zip",
                        mime="application/zip",
                        use_container_width=True
                    )

if __name__ == "__main__":
    main()
