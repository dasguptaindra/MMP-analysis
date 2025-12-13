# app.py
import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
import requests
import tempfile
import os
import sys
from operator import itemgetter
from itertools import combinations
import numpy as np
from pathlib import Path
import time

# Set page config
st.set_page_config(
    page_title="Custom MMP Analyzer",
    page_icon="🧪",
    layout="wide"
)

# Title and description
st.title("🧪 Custom Matched Molecular Pairs Analyzer")
st.markdown("""
Upload your own dataset to perform Matched Molecular Pair (MMP) analysis.
MMPs identify pairs of molecules that differ by a single structural transformation, 
allowing you to analyze how specific chemical changes affect biological activity or properties.
""")

# Custom scaffold finder implementation
class ScaffoldFinder:
    """Custom implementation of scaffold finding functionality"""
    
    @staticmethod
    def get_largest_fragment(mol):
        """Get the largest fragment from a molecule"""
        if mol is None:
            return None
        frags = Chem.GetMolFrags(mol, asMols=True)
        if not frags:
            return mol
        return max(frags, key=lambda x: x.GetNumAtoms())
    
    @staticmethod
    def FragmentMol(mol, maxCuts=1):
        """Simple fragmentation function for MMP analysis"""
        results = []
        if mol is None:
            return results
        
        # Get all bonds that can be cut (single bonds not in rings)
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.SINGLE:
                a1 = bond.GetBeginAtom()
                a2 = bond.GetEndAtom()
                
                # Skip if atoms are in rings
                if a1.IsInRing() or a2.IsInRing():
                    continue
                
                # Skip terminal hydrogens
                if a1.GetAtomicNum() == 1 or a2.GetAtomicNum() == 1:
                    continue
                
                # Create a copy and break the bond
                mol_copy = Chem.RWMol(mol)
                mol_copy.RemoveBond(a1.GetIdx(), a2.GetIdx())
                
                # Convert back to regular molecule
                frag_mol = mol_copy.GetMol()
                results.append((f"{a1.GetIdx()}-{a2.GetIdx()}", frag_mol))
        
        return results

# Initialize the scaffold finder
scaffold_finder = ScaffoldFinder()

# Initialize session state
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'mmp_results' not in st.session_state:
    st.session_state.mmp_results = None

# Sidebar
st.sidebar.header("📁 Data Input")

# Data source selection
data_source = st.sidebar.radio(
    "Choose Data Source:",
    ["Upload CSV File", "Use Example Dataset"],
    index=1
)

example_datasets = {
    "hERG Inhibition": "https://raw.githubusercontent.com/PatWalters/practical_cheminformatics_tutorials/main/data/hERG.csv",
    "Solubility (ESOL)": "https://raw.githubusercontent.com/PatWalters/practical_cheminformatics_tutorials/main/data/ESOL.csv",
    "Lipophilicity": "https://raw.githubusercontent.com/PatWalters/practical_cheminformatics_tutorials/main/data/Lipophilicity.csv"
}

# Data loading section
if data_source == "Upload CSV File":
    st.sidebar.subheader("Upload Your Data")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file",
        type=['csv', 'txt'],
        help="CSV file should contain SMILES, activity/property values, and molecule names"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.uploaded_data = df
            st.sidebar.success(f"✅ Loaded {len(df)} molecules")
            
            # Show column names
            st.sidebar.info(f"Columns: {', '.join(df.columns.tolist())}")
            
        except Exception as e:
            st.sidebar.error(f"Error loading file: {str(e)}")
    
    # Column mapping
    if st.session_state.uploaded_data is not None:
        st.sidebar.subheader("Column Mapping")
        
        df = st.session_state.uploaded_data
        columns = df.columns.tolist()
        
        # Auto-detect columns
        smiles_col = st.sidebar.selectbox(
            "SMILES Column",
            columns,
            index=0 if 'SMILES' in columns else (1 if 'smiles' in [c.lower() for c in columns] else 0)
        )
        
        # Find numeric columns for activity
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        activity_col = st.sidebar.selectbox(
            "Activity/Property Column",
            columns,
            index=0 if numeric_cols else 1
        )
        
        name_col = st.sidebar.selectbox(
            "Name/ID Column (Optional)",
            ['None'] + columns,
            index=0
        )
        
        # Process button
        if st.sidebar.button("🔄 Process Molecules", type="primary", use_container_width=True):
            with st.spinner("Processing molecules..."):
                try:
                    processed_df = df.copy()
                    
                    # Create molecule objects
                    processed_df['mol'] = processed_df[smiles_col].apply(Chem.MolFromSmiles)
                    
                    # Check for invalid molecules
                    invalid_mask = processed_df['mol'].isnull()
                    if invalid_mask.any():
                        invalid_count = invalid_mask.sum()
                        st.warning(f"⚠️ {invalid_count} invalid SMILES found and removed")
                        processed_df = processed_df[~invalid_mask].copy()
                    
                    # Get largest fragment
                    processed_df['mol'] = processed_df['mol'].apply(scaffold_finder.get_largest_fragment)
                    
                    # Rename columns for consistency
                    processed_df = processed_df.rename(columns={
                        smiles_col: 'SMILES',
                        activity_col: 'Activity'
                    })
                    
                    if name_col != 'None':
                        processed_df = processed_df.rename(columns={name_col: 'Name'})
                    else:
                        processed_df['Name'] = [f"Mol_{i+1}" for i in range(len(processed_df))]
                    
                    # Keep only necessary columns
                    processed_df = processed_df[['SMILES', 'Name', 'Activity', 'mol']].copy()
                    
                    st.session_state.processed_data = processed_df
                    st.session_state.analysis_results = None
                    st.session_state.mmp_results = None
                    
                    st.success(f"✅ Successfully processed {len(processed_df)} valid molecules")
                    
                except Exception as e:
                    st.error(f"Error processing molecules: {str(e)}")

else:  # Use Example Dataset
    st.sidebar.subheader("Example Dataset")
    
    selected_dataset = st.sidebar.selectbox(
        "Select Example Dataset:",
        list(example_datasets.keys())
    )
    
    if st.sidebar.button("📥 Load Example Dataset", use_container_width=True):
        with st.spinner(f"Loading {selected_dataset} dataset..."):
            try:
                url = example_datasets[selected_dataset]
                df = pd.read_csv(url)
                st.session_state.uploaded_data = df
                
                # Auto-process example datasets
                processed_df = df.copy()
                
                # Map columns based on dataset
                if selected_dataset == "hERG Inhibition":
                    processed_df = processed_df.rename(columns={
                        'SMILES': 'SMILES',
                        'pIC50': 'Activity',
                        'Name': 'Name'
                    })
                elif selected_dataset == "Solubility (ESOL)":
                    processed_df = processed_df.rename(columns={
                        'smiles': 'SMILES',
                        'measured log solubility in mols per litre': 'Activity',
                        'Compound ID': 'Name'
                    })
                elif selected_dataset == "Lipophilicity":
                    processed_df = processed_df.rename(columns={
                        'smiles': 'SMILES',
                        'exp': 'Activity',
                        'Compound ID': 'Name'
                    })
                
                # Create molecule objects
                processed_df['mol'] = processed_df['SMILES'].apply(Chem.MolFromSmiles)
                processed_df['mol'] = processed_df['mol'].apply(scaffold_finder.get_largest_fragment)
                
                # Remove any invalid molecules
                processed_df = processed_df[processed_df['mol'].notnull()].copy()
                
                st.session_state.processed_data = processed_df
                st.session_state.analysis_results = None
                st.session_state.mmp_results = None
                
                st.sidebar.success(f"✅ Loaded {len(processed_df)} molecules")
                
            except Exception as e:
                st.error(f"Error loading dataset: {str(e)}")

# Analysis Parameters Section
if st.session_state.processed_data is not None:
    st.sidebar.header("⚙️ Analysis Parameters")
    
    # Parameters
    min_transform_occurrence = st.sidebar.slider(
        "Minimum Transform Occurrence",
        min_value=2, max_value=50, value=5,
        help="Minimum number of times a transformation must appear to be considered"
    )
    
    max_cuts = st.sidebar.slider(
        "Maximum Number of Cuts",
        min_value=1, max_value=3, value=1,
        help="Maximum number of bonds to cut when fragmenting molecules"
    )
    
    activity_units = st.sidebar.text_input(
        "Activity Units",
        value="pIC50",
        help="Units for your activity/property values (for display only)"
    )
    
    # Run analysis button
    if st.sidebar.button("🔬 Run MMP Analysis", type="primary", use_container_width=True):
        with st.spinner("Running MMP analysis..."):
            try:
                processed_df = st.session_state.processed_data
                
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Utility functions
                def remove_map_nums(mol):
                    for atom in mol.GetAtoms():
                        atom.SetAtomMapNum(0)
                    return mol
                
                def sort_fragments(mol):
                    frags = list(Chem.GetMolFrags(mol, asMols=True))
                    frags = [remove_map_nums(x) for x in frags]
                    frags.sort(key=lambda x: x.GetNumAtoms(), reverse=True)
                    return frags
                
                # Step 1: Fragment molecules
                status_text.text("Step 1/3: Fragmenting molecules...")
                row_list = []
                total_mols = len(processed_df)
                
                for idx, row in enumerate(processed_df.itertuples()):
                    mol = row.mol
                    frags = scaffold_finder.FragmentMol(mol, maxCuts=max_cuts)
                    
                    for _, frag_mol in frags:
                        pair = sort_fragments(frag_mol)
                        if len(pair) == 2:
                            row_list.append([
                                row.SMILES,
                                Chem.MolToSmiles(pair[0]),
                                Chem.MolToSmiles(pair[1]),
                                row.Name,
                                row.Activity
                            ])
                    
                    progress_bar.progress((idx + 1) / total_mols * 0.33)
                
                if not row_list:
                    st.error("No valid fragments found. Try increasing Max Cuts.")
                    st.stop()
                
                row_df = pd.DataFrame(row_list, columns=["SMILES", "Core", "R_group", "Name", "Activity"])
                
                # Step 2: Find molecular pairs
                status_text.text("Step 2/3: Finding molecular pairs...")
                delta_list = []
                groups = list(row_df.groupby("Core"))
                
                for group_idx, (core, group) in enumerate(groups):
                    if len(group) > 1:
                        mols = group.to_dict('records')
                        for i in range(len(mols)):
                            for j in range(i + 1, len(mols)):
                                a, b = mols[i], mols[j]
                                if a['SMILES'] != b['SMILES']:
                                    # Sort by SMILES for consistency
                                    if a['SMILES'] > b['SMILES']:
                                        a, b = b, a
                                    
                                    delta = b['Activity'] - a['Activity']
                                    transform = f"{a['R_group'].replace('*', '*-')}>>{b['R_group'].replace('*', '*-')}"
                                    delta_list.append([
                                        a['SMILES'], a['Core'], a['R_group'], a['Name'], a['Activity'],
                                        b['SMILES'], b['Core'], b['R_group'], b['Name'], b['Activity'],
                                        transform, delta
                                    ])
                    
                    progress_bar.progress(0.33 + (group_idx + 1) / len(groups) * 0.33)
                
                if not delta_list:
                    st.error("No molecular pairs found.")
                    st.stop()
                
                delta_df = pd.DataFrame(delta_list, columns=[
                    "SMILES_1", "Core_1", "R_group_1", "Name_1", "Activity_1",
                    "SMILES_2", "Core_2", "Rgroup_2", "Name_2", "Activity_2",
                    "Transform", "Delta"
                ])
                
                # Step 3: Aggregate transforms
                status_text.text("Step 3/3: Analyzing transforms...")
                mmp_data = []
                transform_groups = list(delta_df.groupby("Transform"))
                
                for idx, (transform, group) in enumerate(transform_groups):
                    if len(group) >= min_transform_occurrence:
                        deltas = group['Delta'].values
                        mmp_data.append({
                            'Transform': transform,
                            'Count': len(group),
                            'Deltas': deltas,
                            'mean_delta': np.mean(deltas),
                            'std_delta': np.std(deltas),
                            'min_delta': np.min(deltas),
                            'max_delta': np.max(deltas)
                        })
                    
                    progress_bar.progress(0.66 + (idx + 1) / len(transform_groups) * 0.34)
                
                if not mmp_data:
                    st.error(f"No transforms found with occurrence ≥ {min_transform_occurrence}. Try lowering the threshold.")
                    st.stop()
                
                mmp_df = pd.DataFrame(mmp_data)
                mmp_df['idx'] = range(len(mmp_df))
                
                # Add reaction objects for display
                def create_reaction(transform_smarts):
                    try:
                        return AllChem.ReactionFromSmarts(transform_smarts, useSmiles=True)
                    except:
                        return None
                
                mmp_df['rxn_mol'] = mmp_df['Transform'].apply(create_reaction)
                
                # Store results
                st.session_state.analysis_results = {
                    'row_df': row_df,
                    'delta_df': delta_df,
                    'mmp_df': mmp_df
                }
                
                progress_bar.progress(1.0)
                status_text.text("Analysis complete!")
                
                st.success(f"✅ Found {len(mmp_df)} MMPs with occurrence ≥ {min_transform_occurrence}")
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

# Main Display Area
if st.session_state.processed_data is not None:
    # Display dataset info
    processed_df = st.session_state.processed_data
    
    st.header("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Molecules", len(processed_df))
    with col2:
        st.metric("Activity Range", f"{processed_df['Activity'].min():.2f} - {processed_df['Activity'].max():.2f}")
    with col3:
        st.metric("Average Activity", f"{processed_df['Activity'].mean():.2f}")
    with col4:
        unique_cores = len(set([Chem.MolToSmiles(Chem.MurckoDecompose(mol)) for mol in processed_df['mol'] if mol]))
        st.metric("Unique Scaffolds", unique_cores)
    
    # Show data preview
    with st.expander("📋 View Dataset", expanded=False):
        st.dataframe(processed_df[['SMILES', 'Name', 'Activity']], use_container_width=True)
    
    # Show molecular grid
    with st.expander("👁️ View Molecules", expanded=False):
        # Display first 20 molecules
        display_df = processed_df.head(20).copy()
        
        # Create molecule images
        def mol_to_svg(mol):
            if mol:
                return Chem.Draw.MolToImage(mol, size=(200, 200))
            return None
        
        # Display in columns
        cols = st.columns(4)
        for idx, (_, row) in enumerate(display_df.iterrows()):
            with cols[idx % 4]:
                if row['mol']:
                    img = Chem.Draw.MolToImage(row['mol'], size=(200, 200))
                    st.image(img, caption=f"{row['Name']}\nActivity: {row['Activity']:.2f}")
    
    # Display analysis results if available
    if st.session_state.analysis_results is not None:
        results = st.session_state.analysis_results
        mmp_df = results['mmp_df']
        delta_df = results['delta_df']
        
        st.header("🔬 MMP Analysis Results")
        
        # Results summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total MMPs Found", len(mmp_df))
        with col2:
            avg_abs_delta = mmp_df['mean_delta'].abs().mean()
            st.metric("Avg |Δ|", f"{avg_abs_delta:.2f}")
        with col3:
            most_frequent = mmp_df.loc[mmp_df['Count'].idxmax(), 'Count']
            st.metric("Most Frequent", most_frequent)
        with col4:
            max_impact = mmp_df['mean_delta'].abs().max()
            st.metric("Max Impact", f"{max_impact:.2f}")
        
        # Sort options
        st.subheader("📈 Top MMP Transforms")
        
        col1, col2 = st.columns(2)
        with col1:
            sort_by = st.selectbox(
                "Sort by:",
                ["Mean Δ (Decreasing)", "Mean Δ (Increasing)", "Frequency", "Max Impact"]
            )
        
        with col2:
            n_to_show = st.slider("Number to show:", 5, 50, 10)
        
        # Sort based on selection
        if sort_by == "Mean Δ (Decreasing)":
            display_mmp = mmp_df.sort_values('mean_delta', ascending=False)
        elif sort_by == "Mean Δ (Increasing)":
            display_mmp = mmp_df.sort_values('mean_delta', ascending=True)
        elif sort_by == "Frequency":
            display_mmp = mmp_df.sort_values('Count', ascending=False)
        else:  # Max Impact
            display_mmp = mmp_df.copy()
            display_mmp['abs_mean'] = display_mmp['mean_delta'].abs()
            display_mmp = display_mmp.sort_values('abs_mean', ascending=False)
        
        display_mmp = display_mmp.head(n_to_show).reset_index(drop=True)
        
        # Display each MMP
        for idx, row in display_mmp.iterrows():
            with st.expander(
                f"Transform {idx+1}: {row['Transform'][:80]}... | Δ={row['mean_delta']:.2f} ± {row['std_delta']:.2f} | n={row['Count']}",
                expanded=False
            ):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Display reaction
                    if row['rxn_mol']:
                        try:
                            img = Draw.ReactionToImage(row['rxn_mol'])
                            st.image(img, use_column_width=True, caption="Molecular Transformation")
                        except:
                            st.code(row['Transform'])
                    
                    st.write(f"**Transform:**")
                    st.code(row['Transform'])
                    
                    st.write(f"**Statistics:**")
                    st.write(f"- Mean Δ: {row['mean_delta']:.2f}")
                    st.write(f"- Std Dev: {row['std_delta']:.2f}")
                    st.write(f"- Min Δ: {row['min_delta']:.2f}")
                    st.write(f"- Max Δ: {row['max_delta']:.2f}")
                    st.write(f"- Occurrences: {row['Count']}")
                
                with col2:
                    # Plot distribution
                    fig, ax = plt.subplots(figsize=(10, 3))
                    sns.stripplot(x=row['Deltas'], ax=ax, size=10, alpha=0.7, color='blue')
                    sns.boxplot(x=row['Deltas'], ax=ax, width=0.3, color='orange', alpha=0.3)
                    ax.axvline(0, color='red', linestyle='--', alpha=0.7)
                    ax.axvline(row['mean_delta'], color='green', linestyle='-', alpha=0.7)
                    ax.set_xlabel(f'Δ{activity_units}')
                    ax.set_title(f'Distribution of Activity Changes (n={row["Count"]})')
                    ax.set_xlim(-5, 5)
                    st.pyplot(fig)
                    
                    # Show example pairs
                    st.write("**Example Molecule Pairs:**")
                    example_pairs = delta_df[delta_df['Transform'] == row['Transform']].head(3)
                    
                    for _, pair in example_pairs.iterrows():
                        col_a, col_b, col_c = st.columns([3, 1, 3])
                        with col_a:
                            st.write(f"**{pair['Name_1']}**")
                            st.write(f"{activity_units}: {pair['Activity_1']:.2f}")
                            try:
                                mol1 = Chem.MolFromSmiles(pair['SMILES_1'])
                                if mol1:
                                    img1 = Chem.Draw.MolToImage(mol1, size=(150, 150))
                                    st.image(img1)
                            except:
                                pass
                        
                        with col_b:
                            st.write("→")
                            st.write(f"**Δ={pair['Delta']:.2f}**")
                        
                        with col_c:
                            st.write(f"**{pair['Name_2']}**")
                            st.write(f"{activity_units}: {pair['Activity_2']:.2f}")
                            try:
                                mol2 = Chem.MolFromSmiles(pair['SMILES_2'])
                                if mol2:
                                    img2 = Chem.Draw.MolToImage(mol2, size=(150, 150))
                                    st.image(img2)
                            except:
                                pass
                        
                        st.divider()
        
        # Data tables tab
        st.subheader("📋 Detailed Data")
        
        tab1, tab2, tab3 = st.tabs(["MMP Summary", "All Pairs", "Raw Fragments"])
        
        with tab1:
            summary_df = mmp_df[['Transform', 'Count', 'mean_delta', 'std_delta', 'min_delta', 'max_delta']].copy()
            summary_df = summary_df.round(3)
            st.dataframe(summary_df, use_container_width=True)
        
        with tab2:
            pairs_df = delta_df[['SMILES_1', 'Name_1', 'Activity_1', 
                               'SMILES_2', 'Name_2', 'Activity_2', 
                               'Transform', 'Delta']].copy()
            st.dataframe(pairs_df, use_container_width=True)
        
        with tab3:
            st.dataframe(results['row_df'], use_container_width=True)
        
        # Download section
        st.subheader("📥 Download Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = mmp_df[['Transform', 'Count', 'mean_delta', 'std_delta', 'min_delta', 'max_delta']].to_csv(index=False)
            st.download_button(
                label="Download MMP Summary",
                data=csv,
                file_name="mmp_summary.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            csv = delta_df.to_csv(index=False)
            st.download_button(
                label="Download All Pairs",
                data=csv,
                file_name="all_pairs.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col3:
            csv = processed_df[['SMILES', 'Name', 'Activity']].to_csv(index=False)
            st.download_button(
                label="Download Processed Data",
                data=csv,
                file_name="processed_data.csv",
                mime="text/csv",
                use_container_width=True
            )
else:
    # Welcome screen
    st.info("👈 Start by uploading your data or selecting an example dataset from the sidebar.")
    
    # Show example CSV format
    with st.expander("📄 Expected CSV Format"):
        st.markdown("""
        Your CSV file should have the following columns (column names can vary):
        
        | SMILES | Activity | Name |
        |--------|----------|------|
        | CCO | 6.5 | Ethanol |
        | CC(=O)O | 4.2 | Acetic Acid |
        | c1ccccc1 | 5.8 | Benzene |
        
        **Required:**
        - `SMILES` column: Contains molecular structures as SMILES strings
        - `Activity` column: Contains numeric activity/property values
        
        **Optional:**
        - `Name` column: Molecule names or IDs
        - Additional columns will be ignored during analysis
        
        **Note:** The app will automatically map your column names during processing.
        """)
    
    # Quick start guide
    with st.expander("🚀 Quick Start Guide"):
        st.markdown("""
        1. **Choose Data Source** in the sidebar
        2. **Upload your CSV** or select an example dataset
        3. **Map your columns** (if using custom CSV)
        4. **Adjust parameters** as needed
        5. **Click 'Run MMP Analysis'**
        6. **Explore results** in the main panel
        
        **Tips for Best Results:**
        - Ensure all SMILES are valid
        - Activity values should be numeric (higher = better activity)
        - Start with at least 50-100 molecules for meaningful MMPs
        - Adjust "Minimum Transform Occurrence" based on dataset size
        """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>🧪 Matched Molecular Pairs Analyzer | 
        <a href='https://doi.org/10.1021/ci900450m' target='_blank'>Hussain & Rea, 2010</a> | 
        Customizable Version</p>
    </div>
    """,
    unsafe_allow_html=True
)
