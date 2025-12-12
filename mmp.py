import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from operator import itemgetter
from itertools import combinations
import tempfile
import os
import io
import base64
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Function to remove atom map numbers
def remove_map_nums(mol):
    for atm in mol.GetAtoms():
        atm.SetAtomMapNum(0)
    return mol

# Function to sort fragments by size
def sort_fragments(mol):
    frag_list = list(Chem.GetMolFrags(mol, asMols=True))
    [remove_map_nums(x) for x in frag_list]
    frag_num_atoms_list = [(x.GetNumAtoms(), x) for x in frag_list]
    frag_num_atoms_list.sort(key=itemgetter(0), reverse=True)
    return [x[1] for x in frag_num_atoms_list]

# Simple scaffold finder (mimicking the one used in the notebook)
def FragmentMol(mol, maxCuts=1):
    results = []
    for bond in mol.GetBonds():
        # Create a copy of the molecule
        mol_copy = Chem.RWMol(mol)
        # Remove the bond (single cut)
        mol_copy.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        # Convert back to regular molecule
        fragmented = mol_copy.GetMol()
        # Add dummy atoms to mark the cut points
        fragmented = AllChem.ReplaceSidechains(fragmented, mol)
        results.append((bond.GetIdx(), fragmented))
    return results

def get_largest_fragment(mol):
    frags = Chem.GetMolFrags(mol, asMols=True)
    if frags:
        return max(frags, key=lambda x: x.GetNumAtoms())
    return mol

# Function to create MMP analysis
def perform_mmp_analysis(df, min_transform_occurrence=5):
    # Create molecule objects
    df['mol'] = df['SMILES'].apply(Chem.MolFromSmiles)
    df['mol'] = df['mol'].apply(get_largest_fragment)
    
    # Decompose molecules to scaffolds and sidechains
    row_list = []
    for smiles, name, pIC50, mol in df[['SMILES', 'Name', 'pIC50', 'mol']].values:
        frag_list = FragmentMol(mol, maxCuts=1)
        for _, frag_mol in frag_list:
            pair_list = sort_fragments(frag_mol)
            if len(pair_list) == 2:  # Should always be 2 for single cuts
                tmp_list = [smiles] + [Chem.MolToSmiles(x) for x in pair_list] + [name, pIC50]
                row_list.append(tmp_list)
    
    row_df = pd.DataFrame(row_list, columns=["SMILES", "Core", "R_group", "Name", "pIC50"])
    
    # Collect pairs with the same scaffold
    delta_list = []
    for core, group in row_df.groupby("Core"):
        if len(group) > 2:
            for a, b in combinations(range(len(group)), 2):
                reagent_a = group.iloc[a]
                reagent_b = group.iloc[b]
                if reagent_a.SMILES == reagent_b.SMILES:
                    continue
                
                # Sort by SMILES for canonical ordering
                reagent_a, reagent_b = sorted([reagent_a, reagent_b], key=lambda x: x.SMILES)
                delta = reagent_b.pIC50 - reagent_a.pIC50
                
                delta_list.append(list(reagent_a.values) + list(reagent_b.values) + 
                                  [f"{reagent_a.R_group.replace('*', '*-')}>>{reagent_b.R_group.replace('*', '*-')}", delta])
    
    cols = ["SMILES_1", "Core_1", "R_group_1", "Name_1", "pIC50_1",
            "SMILES_2", "Core_2", "Rgroup_2", "Name_2", "pIC50_2",
            "Transform", "Delta"]
    
    delta_df = pd.DataFrame(delta_list, columns=cols)
    
    # Collect frequently occurring pairs
    mmp_list = []
    for transform, group in delta_df.groupby("Transform"):
        if len(group) > min_transform_occurrence:
            mmp_list.append([transform, len(group), group.Delta.values])
    
    mmp_df = pd.DataFrame(mmp_list, columns=["Transform", "Count", "Deltas"])
    mmp_df['idx'] = range(len(mmp_df))
    mmp_df['mean_delta'] = [x.mean() for x in mmp_df.Deltas]
    
    # Create reaction molecules for display
    def create_rxn_mol(transform):
        try:
            # Remove the *- prefix for reaction creation
            transform_clean = transform.replace('*-', '')
            return AllChem.ReactionFromSmarts(transform_clean, useSmiles=True)
        except:
            return None
    
    mmp_df['rxn_mol'] = mmp_df['Transform'].apply(create_rxn_mol)
    
    # Add index column to delta_df for linking
    transform_dict = dict(zip(mmp_df["Transform"], mmp_df["idx"]))
    delta_df['idx'] = delta_df['Transform'].map(transform_dict)
    
    return row_df, delta_df, mmp_df

# Function to display molecule
def display_molecule(smiles, title=""):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        img = Draw.MolToImage(mol, size=(300, 200))
        st.image(img, caption=title, use_column_width=True)
    else:
        st.error(f"Could not create molecule from SMILES: {smiles}")

# Function to display reaction
def display_reaction(rxn):
    if rxn:
        img = Draw.ReactionToImage(rxn)
        st.image(img, use_column_width=True)
    else:
        st.warning("Could not display reaction")

# Streamlit app
def main():
    st.set_page_config(page_title="MMP Analysis Tool", layout="wide")
    
    st.title("📊 Matched Molecular Pairs (MMP) Analysis")
    st.markdown("""
    This tool performs Matched Molecular Pairs analysis to identify structural transformations
    that affect biological activity (pIC50).
    
    Upload your CSV file with columns: **SMILES**, **Name**, **pIC50**
    """)
    
    # Sidebar for file upload and parameters
    with st.sidebar:
        st.header("📁 Upload Data")
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        st.header("⚙️ Analysis Parameters")
        min_occurrence = st.slider(
            "Minimum Transform Occurrences",
            min_value=2,
            max_value=20,
            value=5,
            help="Minimum number of times a transform must occur to be included"
        )
        
        st.header("ℹ️ About")
        st.markdown("""
        This app performs Matched Molecular Pair (MMP) analysis:
        1. Fragments molecules at single bonds
        2. Identifies pairs sharing the same scaffold
        3. Calculates ΔpIC50 for each pair
        4. Finds frequently occurring transformations
        """)
    
    # Main content area
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            
            # Check required columns
            required_cols = ['SMILES', 'Name', 'pIC50']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
                st.info("Please ensure your CSV has columns: SMILES, Name, pIC50")
            else:
                # Display data preview
                st.subheader("📋 Data Preview")
                st.dataframe(df.head(10))
                
                # Show basic statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Number of Compounds", len(df))
                with col2:
                    st.metric("pIC50 Range", f"{df['pIC50'].min():.2f} - {df['pIC50'].max():.2f}")
                with col3:
                    st.metric("Average pIC50", f"{df['pIC50'].mean():.2f}")
                
                # Perform MMP analysis
                with st.spinner("Performing MMP analysis..."):
                    row_df, delta_df, mmp_df = perform_mmp_analysis(df, min_transform_occurrence)
                
                # Display results
                st.subheader("🔬 MMP Analysis Results")
                
                # Tab layout for different views
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📈 Transform Summary", 
                    "🔍 Detailed Pairs", 
                    "🧪 Decompositions",
                    "📊 Visualizations"
                ])
                
                with tab1:
                    st.subheader("Frequently Occurring Transformations")
                    
                    if len(mmp_df) > 0:
                        # Sort by mean delta
                        mmp_df_sorted = mmp_df.sort_values('mean_delta', ascending=False)
                        
                        for idx, row in mmp_df_sorted.iterrows():
                            with st.expander(f"Transform: {row['Transform']} (Count: {row['Count']}, ΔpIC50: {row['mean_delta']:.3f})"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write("**Statistics:**")
                                    st.write(f"- Occurrences: {row['Count']}")
                                    st.write(f"- Mean ΔpIC50: {row['mean_delta']:.3f}")
                                    st.write(f"- Standard Deviation: {np.std(row['Deltas']):.3f}")
                                    st.write(f"- Min ΔpIC50: {min(row['Deltas']):.3f}")
                                    st.write(f"- Max ΔpIC50: {max(row['Deltas']):.3f}")
                                
                                with col2:
                                    if row['rxn_mol'] is not None:
                                        st.write("**Reaction:**")
                                        display_reaction(row['rxn_mol'])
                                    else:
                                        st.write("Could not display reaction visualization")
                    else:
                        st.warning(f"No transformations found with minimum occurrence of {min_occurrence}. Try lowering the threshold.")
                
                with tab2:
                    st.subheader("Detailed Molecular Pairs")
                    
                    if len(delta_df) > 0:
                        # Filter to show only pairs with transformations in mmp_df
                        filtered_delta = delta_df[~delta_df['idx'].isna()]
                        
                        if len(filtered_delta) > 0:
                            st.dataframe(filtered_delta.head(50))
                            
                            # Allow user to view specific pair details
                            pair_idx = st.selectbox(
                                "Select a pair to view details",
                                options=filtered_delta.index[:20],
                                format_func=lambda x: f"{filtered_delta.loc[x, 'Name_1']} -> {filtered_delta.loc[x, 'Name_2']}"
                            )
                            
                            if pair_idx:
                                pair = filtered_delta.loc[pair_idx]
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.write("**Molecule 1:**")
                                    st.write(f"Name: {pair['Name_1']}")
                                    st.write(f"SMILES: {pair['SMILES_1']}")
                                    st.write(f"pIC50: {pair['pIC50_1']:.3f}")
                                    display_molecule(pair['SMILES_1'], "Molecule 1")
                                
                                with col2:
                                    st.write("**Molecule 2:**")
                                    st.write(f"Name: {pair['Name_2']}")
                                    st.write(f"SMILES: {pair['SMILES_2']}")
                                    st.write(f"pIC50: {pair['pIC50_2']:.3f}")
                                    display_molecule(pair['SMILES_2'], "Molecule 2")
                                
                                st.write(f"**ΔpIC50:** {pair['Delta']:.3f}")
                                st.write(f"**Transform:** {pair['Transform']}")
                        else:
                            st.info("No detailed pairs available. Try adjusting the minimum occurrence threshold.")
                    else:
                        st.info("No molecular pairs found. The dataset may be too small or diverse.")
                
                with tab3:
                    st.subheader("Molecule Decompositions")
                    st.write("Showing first 20 decompositions:")
                    st.dataframe(row_df.head(20))
                
                with tab4:
                    st.subheader("Visualizations")
                    
                    if len(mmp_df) > 0:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Histogram of mean deltas
                            fig, ax = plt.subplots(figsize=(8, 4))
                            ax.hist(mmp_df['mean_delta'], bins=20, edgecolor='black', alpha=0.7)
                            ax.set_xlabel('Mean ΔpIC50')
                            ax.set_ylabel('Frequency')
                            ax.set_title('Distribution of Transform Effects')
                            st.pyplot(fig)
                        
                        with col2:
                            # Scatter plot of count vs mean delta
                            fig, ax = plt.subplots(figsize=(8, 4))
                            ax.scatter(mmp_df['Count'], mmp_df['mean_delta'], alpha=0.6)
                            ax.set_xlabel('Transform Count')
                            ax.set_ylabel('Mean ΔpIC50')
                            ax.set_title('Transform Frequency vs Effect')
                            st.pyplot(fig)
                        
                        # Show delta distribution for top transformations
                        st.subheader("ΔpIC50 Distribution for Top Transformations")
                        
                        top_n = min(5, len(mmp_df))
                        top_transforms = mmp_df.sort_values('mean_delta', ascending=False).head(top_n)
                        
                        for idx, row in top_transforms.iterrows():
                            with st.expander(f"Distribution for: {row['Transform']}"):
                                fig, ax = plt.subplots(figsize=(10, 3))
                                ax.hist(row['Deltas'], bins=15, edgecolor='black', alpha=0.7)
                                ax.axvline(row['mean_delta'], color='red', linestyle='--', label=f'Mean: {row["mean_delta"]:.3f}')
                                ax.set_xlabel('ΔpIC50')
                                ax.set_ylabel('Frequency')
                                ax.set_title(f'ΔpIC50 Distribution (n={row["Count"]})')
                                ax.legend()
                                st.pyplot(fig)
                    else:
                        st.info("No visualizations available. Need transformations with sufficient occurrences.")
                
                # Download buttons
                st.subheader("💾 Download Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    csv1 = row_df.to_csv(index=False)
                    st.download_button(
                        label="Download Decompositions",
                        data=csv1,
                        file_name="mmp_decompositions.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    csv2 = delta_df.to_csv(index=False)
                    st.download_button(
                        label="Download Molecular Pairs",
                        data=csv2,
                        file_name="mmp_pairs.csv",
                        mime="text/csv"
                    )
                
                with col3:
                    csv3 = mmp_df[['Transform', 'Count', 'mean_delta']].to_csv(index=False)
                    st.download_button(
                        label="Download Transform Summary",
                        data=csv3,
                        file_name="mmp_transforms.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please check your CSV format and try again.")
    
    else:
        # Display example and instructions
        st.info("👈 Please upload a CSV file to begin analysis")
        
        # Show example data format
        st.subheader("📝 Expected CSV Format")
        example_data = {
            'SMILES': ['CCO', 'CCN', 'CC(=O)O'],
            'Name': ['Ethanol', 'Methylamine', 'Acetic acid'],
            'pIC50': [5.0, 6.2, 4.8]
        }
        example_df = pd.DataFrame(example_data)
        st.dataframe(example_df)
        
        st.subheader("📋 Requirements")
        st.markdown("""
        1. CSV file with headers
        2. Required columns: **SMILES**, **Name**, **pIC50**
        3. SMILES should be valid chemical structures
        4. pIC50 should be numeric values
        5. Minimum 10-20 compounds recommended for meaningful analysis
        """)
        
        # Provide download link for example data
        csv = example_df.to_csv(index=False)
        st.download_button(
            label="Download Example CSV",
            data=csv,
            file_name="example_mmp_data.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
