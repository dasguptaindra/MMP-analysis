# app_simple.py - Simplified version without external dependencies
import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdMolDescriptors
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
import requests
import tempfile
import os
import sys
from pathlib import Path
from operator import itemgetter
from itertools import combinations
import numpy as np

# Set page config
st.set_page_config(
    page_title="MMP Analyzer",
    page_icon="🧪",
    layout="wide"
)

# Custom scaffold finder implementation
class SimpleScaffoldFinder:
    """Simple implementation of scaffold finding functionality"""
    
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
        """Simple fragmentation function"""
        results = []
        if mol is None:
            return results
        
        # Get all bonds that can be cut (single bonds not in rings)
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.SINGLE:
                a1 = bond.GetBeginAtom()
                a2 = bond.GetEndAtom()
                
                # Check if atoms are in rings
                if a1.IsInRing() or a2.IsInRing():
                    continue
                
                # Create a copy and break the bond
                mol_copy = Chem.RWMol(mol)
                mol_copy.RemoveBond(a1.GetIdx(), a2.GetIdx())
                
                # Add map numbers for tracking
                for atom in mol_copy.GetAtoms():
                    atom.SetAtomMapNum(atom.GetIdx() + 1)
                
                # Convert back to regular molecule
                frag_mol = mol_copy.GetMol()
                results.append((f"{a1.GetIdx()}-{a2.GetIdx()}", frag_mol))
        
        return results

# Initialize the scaffold finder
scaffold_finder = SimpleScaffoldFinder()

# Title
st.title("🧪 Matched Molecular Pairs Analysis")

# Initialize session state
for key in ['data_loaded', 'analysis_complete', 'df', 'mmp_df', 'delta_df']:
    if key not in st.session_state:
        st.session_state[key] = False if key in ['data_loaded', 'analysis_complete'] else None

# Sidebar
st.sidebar.header("Parameters")

# Load data
if st.sidebar.button("📥 Load Data"):
    with st.spinner("Loading hERG data..."):
        try:
            url = "https://raw.githubusercontent.com/PatWalters/practical_cheminformatics_tutorials/main/data/hERG.csv"
            df = pd.read_csv(url)
            
            # Process molecules
            df['mol'] = df['SMILES'].apply(Chem.MolFromSmiles)
            df['mol'] = df['mol'].apply(scaffold_finder.get_largest_fragment)
            df = df[df['mol'].notnull()].copy()
            
            st.session_state.df = df
            st.session_state.data_loaded = True
            st.sidebar.success(f"Loaded {len(df)} molecules")
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Main analysis function
def run_mmp_analysis(df, min_count=5):
    """Run MMP analysis on the dataframe"""
    
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
    row_list = []
    for _, row in df.iterrows():
        mol = row['mol']
        frags = scaffold_finder.FragmentMol(mol, maxCuts=1)
        
        for _, frag_mol in frags:
            pair = sort_fragments(frag_mol)
            if len(pair) == 2:
                row_list.append([
                    row['SMILES'],
                    Chem.MolToSmiles(pair[0]),
                    Chem.MolToSmiles(pair[1]),
                    row['Name'],
                    row['pIC50']
                ])
    
    row_df = pd.DataFrame(row_list, columns=["SMILES", "Core", "R_group", "Name", "pIC50"])
    
    # Step 2: Find pairs
    delta_list = []
    for core, group in row_df.groupby("Core"):
        if len(group) > 1:
            mols = group.to_dict('records')
            for i in range(len(mols)):
                for j in range(i+1, len(mols)):
                    a, b = mols[i], mols[j]
                    if a['SMILES'] != b['SMILES']:
                        delta = b['pIC50'] - a['pIC50']
                        transform = f"{a['R_group'].replace('*', '*-')}>>{b['R_group'].replace('*', '*-')}"
                        delta_list.append([a['SMILES'], a['Core'], a['R_group'], a['Name'], a['pIC50'],
                                         b['SMILES'], b['Core'], b['R_group'], b['Name'], b['pIC50'],
                                         transform, delta])
    
    delta_df = pd.DataFrame(delta_list, columns=[
        "SMILES_1", "Core_1", "R_group_1", "Name_1", "pIC50_1",
        "SMILES_2", "Core_2", "Rgroup_2", "Name_2", "pIC50_2",
        "Transform", "Delta"
    ])
    
    # Step 3: Aggregate transforms
    mmp_data = []
    for transform, group in delta_df.groupby("Transform"):
        if len(group) >= min_count:
            deltas = group['Delta'].values
            mmp_data.append({
                'Transform': transform,
                'Count': len(group),
                'Deltas': deltas,
                'mean_delta': np.mean(deltas),
                'std_delta': np.std(deltas)
            })
    
    mmp_df = pd.DataFrame(mmp_data)
    mmp_df['idx'] = range(len(mmp_df))
    
    # Add reaction objects for display
    def create_reaction(transform_smarts):
        try:
            return AllChem.ReactionFromSmarts(transform_smarts, useSmiles=True)
        except:
            return None
    
    mmp_df['rxn_mol'] = mmp_df['Transform'].apply(create_reaction)
    
    return row_df, delta_df, mmp_df

# Run analysis
if st.session_state.data_loaded:
    min_count = st.sidebar.slider("Min Transform Count", 2, 20, 5)
    
    if st.sidebar.button("🔬 Analyze"):
        with st.spinner("Running MMP analysis..."):
            try:
                row_df, delta_df, mmp_df = run_mmp_analysis(st.session_state.df, min_count)
                st.session_state.row_df = row_df
                st.session_state.delta_df = delta_df
                st.session_state.mmp_df = mmp_df
                st.session_state.analysis_complete = True
                st.success(f"Found {len(mmp_df)} MMPs")
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

# Display results
if st.session_state.analysis_complete:
    st.header("Results")
    
    # Statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Molecules", len(st.session_state.df))
    with col2:
        st.metric("MMP Transforms", len(st.session_state.mmp_df))
    with col3:
        avg_delta = st.session_state.mmp_df['mean_delta'].mean()
        st.metric("Avg ΔpIC50", f"{avg_delta:.2f}")
    
    # Display top transforms
    st.subheader("Top MMP Transforms")
    
    sort_asc = st.checkbox("Sort by Decreasing Activity", value=True)
    mmp_sorted = st.session_state.mmp_df.sort_values('mean_delta', ascending=not sort_asc)
    
    for idx, row in mmp_sorted.head(10).iterrows():
        with st.expander(f"{row['Transform'][:60]}... (Δ={row['mean_delta']:.2f}, n={row['Count']})"):
            col1, col2 = st.columns(2)
            
            with col1:
                if row['rxn_mol']:
                    try:
                        img = Draw.ReactionToImage(row['rxn_mol'])
                        st.image(img, use_column_width=True)
                    except:
                        st.write("Reaction:", row['Transform'])
            
            with col2:
                # Plot distribution
                fig, ax = plt.subplots(figsize=(8, 2))
                sns.stripplot(x=row['Deltas'], ax=ax, size=10)
                ax.axvline(0, color='red', linestyle='--')
                ax.set_xlabel('ΔpIC50')
                ax.set_title(f'Distribution (n={row["Count"]})')
                st.pyplot(fig)
    
    # Data tables
    st.subheader("Data Tables")
    tab1, tab2 = st.tabs(["MMP Summary", "All Pairs"])
    
    with tab1:
        st.dataframe(mmp_sorted[['Transform', 'Count', 'mean_delta', 'std_delta']], use_container_width=True)
    
    with tab2:
        st.dataframe(st.session_state.delta_df, use_container_width=True)

else:
    # Instructions
    st.info("""
    ### Instructions:
    1. Click **"Load Data"** in the sidebar to load the hERG dataset
    2. Adjust parameters as needed
    3. Click **"Analyze"** to run the MMP analysis
    
    The app will identify molecular pairs that differ by single transformations
    and show how these transformations affect hERG inhibition (pIC50).
    """)

# Footer
st.markdown("---")
st.markdown("*MMP Analysis Tool v1.0*")
