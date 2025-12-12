import streamlit as st
import pandas as pd
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
import sys
import subprocess

# Try to import RDKit with error handling
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw
    from rdkit.Chem.Draw import rdMolDraw2D
    RDKIT_AVAILABLE = True
except ImportError as e:
    st.error(f"RDKit import error: {e}")
    RDKIT_AVAILABLE = False

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

# Check if RDKit is available
if not RDKIT_AVAILABLE:
    st.error("""
    **RDKit is not available!**
    
    Please install RDKit to use this app. You can install it with:
    
    ```bash
    pip install rdkit-pypi
    ```
    
    Or if you're using conda:
    
    ```bash
    conda install -c conda-forge rdkit
    ```
    """)
    st.stop()

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
CFc1ccc(cc1)S(=O)(=O)N,Compound3,6.9
...""")

# Parameters in sidebar
st.sidebar.header("⚙️ Parameters")
min_transform_occurrence = st.sidebar.slider(
    "Minimum transform occurrences",
    min_value=1,
    max_value=20,
    value=2,
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

# Simple fragmentation method for common substituents
def get_mmp_fragments(mol):
    """Simple fragmentation method for MMP analysis"""
    fragments = []
    
    try:
        # Find all single bonds that connect ring systems to substituents
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
                a1 = bond.GetBeginAtom()
                a2 = bond.GetEndAtom()
                
                # Check if this is a bond between ring and non-ring atoms
                a1_in_ring = a1.IsInRing()
                a2_in_ring = a2.IsInRing()
                
                if a1_in_ring != a2_in_ring:
                    # This bond connects a ring to a substituent
                    mol_copy = Chem.RWMol(mol)
                    
                    # Create a copy to modify
                    bond_idx = bond.GetIdx()
                    
                    # Get the atoms
                    a1_idx = bond.GetBeginAtomIdx()
                    a2_idx = bond.GetEndAtomIdx()
                    
                    # Determine which atom is the attachment point (the one in the ring)
                    if a1_in_ring:
                        attach_point_idx = a1_idx
                        substituent_idx = a2_idx
                    else:
                        attach_point_idx = a2_idx
                        substituent_idx = a1_idx
                    
                    # Create scaffold with wildcard
                    scaffold_mol = Chem.RWMol(mol)
                    
                    # Remove the substituent atom and its connected atoms
                    # First, get all atoms in the substituent
                    visited = set()
                    to_visit = [substituent_idx]
                    
                    while to_visit:
                        current = to_visit.pop()
                        if current not in visited:
                            visited.add(current)
                            atom = mol.GetAtomWithIdx(current)
                            for neighbor in atom.GetNeighbors():
                                if neighbor.GetIdx() != attach_point_idx:
                                    to_visit.append(neighbor.GetIdx())
                    
                    # Remove substituent atoms from scaffold
                    atoms_to_remove = sorted(list(visited), reverse=True)
                    for atom_idx in atoms_to_remove:
                        try:
                            scaffold_mol.RemoveAtom(atom_idx)
                        except:
                            pass
                    
                    # Add wildcard to attachment point
                    if scaffold_mol.GetNumAtoms() > 0:
                        scaffold_atom = scaffold_mol.GetAtomWithIdx(attach_point_idx)
                        scaffold_atom.SetAtomMapNum(1)
                        
                        # Create substituent with wildcard
                        substituent_mol = Chem.RWMol()
                        
                        # Create atom mapping
                        atom_map = {}
                        
                        # Add substituent atoms
                        for atom_idx in visited:
                            atom = mol.GetAtomWithIdx(atom_idx)
                            new_atom = Chem.Atom(atom.GetAtomicNum())
                            new_atom.SetFormalCharge(atom.GetFormalCharge())
                            new_atom.SetIsAromatic(atom.GetIsAromatic())
                            new_idx = substituent_mol.AddAtom(new_atom)
                            atom_map[atom_idx] = new_idx
                        
                        # Add bonds
                        for atom_idx in visited:
                            atom = mol.GetAtomWithIdx(atom_idx)
                            for neighbor in atom.GetNeighbors():
                                neighbor_idx = neighbor.GetIdx()
                                if neighbor_idx in visited:
                                    bond = mol.GetBondBetweenAtoms(atom_idx, neighbor_idx)
                                    if bond:
                                        substituent_mol.AddBond(
                                            atom_map[atom_idx], 
                                            atom_map[neighbor_idx], 
                                            bond.GetBondType()
                                        )
                        
                        # Add wildcard for attachment
                        if attach_point_idx in atom_map:
                            attach_atom = substituent_mol.GetAtomWithIdx(atom_map[attach_point_idx])
                            attach_atom.SetAtomMapNum(2)
                        
                        # Only keep if both fragments have atoms
                        if scaffold_mol.GetNumAtoms() > 0 and substituent_mol.GetNumAtoms() > 0:
                            # Convert to regular molecules
                            scaffold = scaffold_mol.GetMol()
                            substituent = substituent_mol.GetMol()
                            
                            # Clean up
                            Chem.SanitizeMol(scaffold)
                            Chem.SanitizeMol(substituent)
                            
                            fragments.append((scaffold, substituent))
    
    except Exception as e:
        st.warning(f"Fragmentation error: {e}")
    
    return fragments

def get_largest_fragment(mol):
    """Get the largest fragment from a molecule"""
    try:
        frags = Chem.GetMolFrags(mol, asMols=True)
        if frags:
            return max(frags, key=lambda x: x.GetNumAtoms())
        return mol
    except:
        return mol

# Plotting functions
def rxn_to_image(rxn_smarts, width=300, height=150):
    """Convert reaction SMARTS to PIL Image"""
    try:
        # Create reaction from SMARTS
        rxn = AllChem.ReactionFromSmarts(rxn_smarts, useSmiles=True)
        img = Draw.ReactionToImage(rxn, subImgSize=(width, height))
        return img
    except:
        # Create a placeholder image
        img = Image.new('RGB', (width, height), color='white')
        return img

def mol_to_image(mol, width=200, height=200):
    """Convert RDKit molecule to PIL Image"""
    try:
        img = Draw.MolToImage(mol, size=(width, height))
        return img
    except:
        # Create a placeholder image
        img = Image.new('RGB', (width, height), color='white')
        return img

def create_stripplot(deltas, figsize=(4, 1.5)):
    """Create a stripplot for delta distribution"""
    fig, ax = plt.subplots(figsize=figsize)
    try:
        if len(deltas) > 0:
            sns.stripplot(x=deltas, ax=ax, jitter=0.3, alpha=0.7, s=8, color='steelblue')
        ax.axvline(0, ls="--", c="red", alpha=0.7)
        
        # Set appropriate xlim based on data
        if len(deltas) > 0:
            x_range = max(abs(min(deltas)), abs(max(deltas))) * 1.1
            ax.set_xlim(-x_range, x_range)
        else:
            ax.set_xlim(-5, 5)
            
        ax.set_xlabel("ΔpIC50")
        ax.set_ylabel("")
        ax.set_yticks([])
        plt.tight_layout()
    except Exception as e:
        ax.text(0.5, 0.5, "Error creating plot", ha='center', va='center')
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
        
        try:
            mol = Chem.MolFromSmiles(row[smiles_col])
            if mol:
                img = Draw.MolToImage(mol, size=(250, 250))
                axes[row_idx][col_idx].imshow(img)
                title = f"{row[id_col]}\n{value_col}: {row[value_col]:.2f}"
                axes[row_idx][col_idx].set_title(title, fontsize=9)
            else:
                axes[row_idx][col_idx].text(0.5, 0.5, "Invalid SMILES", 
                                           ha='center', va='center')
        except:
            axes[row_idx][col_idx].text(0.5, 0.5, "Error", ha='center', va='center')
        
        axes[row_idx][col_idx].axis('off')
    
    # Hide empty subplots
    for idx in range(len(compounds_df), n_rows * n_cols):
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        axes[row_idx][col_idx].axis('off')
    
    plt.tight_layout()
    return fig

# Main analysis function
def run_mmp_analysis(df, min_occurrences=2):
    """Main MMP analysis pipeline"""
    
    # Process molecules
    df['mol'] = df['SMILES'].apply(Chem.MolFromSmiles)
    valid_df = df[df['mol'].notna()].copy()
    
    if len(valid_df) == 0:
        st.error("No valid molecules found in the dataset.")
        return None, None, None, None
    
    st.info(f"✅ Found {len(valid_df)} valid molecules out of {len(df)} total.")
    
    # Decompose molecules to scaffolds and sidechains
    st.info("🔬 Decomposing molecules...")
    progress_bar = st.progress(0)
    
    row_list = []
    for idx, (smiles, name, pIC50, mol) in enumerate(valid_df[['SMILES', 'Name', 'pIC50', 'mol']].values):
        try:
            # Get fragments using our custom method
            fragments = get_mmp_fragments(mol)
            
            for scaffold_mol, substituent_mol in fragments:
                # Get SMILES for fragments
                scaffold_smiles = Chem.MolToSmiles(scaffold_mol)
                substituent_smiles = Chem.MolToSmiles(substituent_mol)
                
                # Add to results
                row_list.append([smiles, scaffold_smiles, substituent_smiles, name, pIC50])
        
        except Exception as e:
            st.warning(f"Error processing molecule {name}: {e}")
        
        progress_bar.progress((idx + 1) / len(valid_df))
    
    if not row_list:
        st.error("No valid fragments generated. This could be because:")
        st.error("1. Molecules don't have clear ring-substituent bonds")
        st.error("2. All molecules are too similar")
        st.error("3. Try using a different dataset with more diverse substituents")
        return None, None, None, None
    
    row_df = pd.DataFrame(row_list, columns=["SMILES", "Core", "R_group", "Name", "pIC50"])
    st.info(f"✅ Generated {len(row_df)} fragment pairs.")
    
    # Collect pairs with same scaffold
    st.info("🤝 Finding molecular pairs...")
    delta_list = []
    
    # Group by scaffold and find pairs
    for core_smiles, group in row_df.groupby("Core"):
        if len(group) > 1:
            # Get all compounds with this scaffold
            compounds = group.to_dict('records')
            
            # Create all possible pairs
            for i in range(len(compounds)):
                for j in range(i+1, len(compounds)):
                    comp_a = compounds[i]
                    comp_b = compounds[j]
                    
                    if comp_a['SMILES'] != comp_b['SMILES']:
                        # Calculate delta pIC50
                        delta = comp_b['pIC50'] - comp_a['pIC50']
                        
                        # Create transform string (replace * with wildcard)
                        transform = f"{comp_a['R_group'].replace('*', '*-')}>>{comp_b['R_group'].replace('*', '*-')}"
                        
                        delta_list.append([
                            comp_a['SMILES'], comp_a['Core'], comp_a['R_group'], comp_a['Name'], comp_a['pIC50'],
                            comp_b['SMILES'], comp_b['Core'], comp_b['R_group'], comp_b['Name'], comp_b['pIC50'],
                            transform, delta
                        ])
    
    if not delta_list:
        st.error("No molecular pairs found. Try:")
        st.error("1. Lowering the minimum occurrence threshold")
        st.error("2. Using a dataset with more diverse compounds sharing common scaffolds")
        return None, None, None, None
    
    cols = ["SMILES_1", "Core_1", "R_group_1", "Name_1", "pIC50_1",
           "SMILES_2", "Core_2", "Rgroup_1", "Name_2", "pIC50_2",
           "Transform", "Delta"]
    delta_df = pd.DataFrame(delta_list, columns=cols)
    st.info(f"✅ Found {len(delta_df)} molecular pairs.")
    
    # Collect frequently occurring pairs
    st.info("📊 Analyzing transformations...")
    mmp_list = []
    transform_to_idx = {}  # Dictionary to map transform to index
    
    for idx, (transform, group) in enumerate(delta_df.groupby("Transform")):
        if len(group) >= min_occurrences:
            mmp_list.append([transform, len(group), group.Delta.values])
            transform_to_idx[transform] = idx
    
    if not mmp_list:
        st.warning(f"No transformations found with at least {min_occurrences} occurrences.")
        st.warning("Try lowering the minimum occurrence threshold.")
        return valid_df, row_df, delta_df, pd.DataFrame()
    
    mmp_df = pd.DataFrame(mmp_list, columns=["Transform", "Count", "Deltas"])
    mmp_df['idx'] = range(len(mmp_df))
    mmp_df['mean_delta'] = [x.mean() for x in mmp_df.Deltas]
    mmp_df['std_delta'] = [x.std() for x in mmp_df.Deltas]
    
    st.info(f"✅ Found {len(mmp_df)} significant transformations.")
    
    return valid_df, row_df, delta_df, mmp_df

# Main app logic
if uploaded_file is not None:
    try:
        # Load data
        df = pd.read_csv(uploaded_file)
        
        # Check required columns
        required_cols = ['SMILES', 'Name', 'pIC50']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            st.info(f"Available columns: {list(df.columns)}")
            st.stop()
        
        # Display data preview
        st.header("📊 Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        st.write(f"Total compounds: {len(df)}")
        
        # Basic statistics
        st.subheader("📈 Basic Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean pIC50", f"{df['pIC50'].mean():.2f}")
        with col2:
            st.metric("Min pIC50", f"{df['pIC50'].min():.2f}")
        with col3:
            st.metric("Max pIC50", f"{df['pIC50'].max():.2f}")
        
        # Run MMP analysis
        with st.spinner("Running MMP analysis..."):
            result = run_mmp_analysis(df, min_transform_occurrence)
            
            if result[0] is None:
                st.error("Analysis failed. Please check your data and parameters.")
                st.stop()
            
            df_processed, row_df, delta_df, mmp_df = result
        
        if len(mmp_df) == 0:
            st.warning("No significant transformations found.")
            st.info("Try:")
            st.info("1. Lowering the minimum occurrence threshold")
            st.info("2. Uploading a dataset with more compounds sharing common scaffolds")
            st.info("3. Checking that your compounds have diverse substituents on common rings")
        else:
            st.success(f"✅ Analysis complete! Found {len(mmp_df)} significant transformations")
            
            # Display results in tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Top Transforms", "📉 Bottom Transforms", "🔍 All Transforms", "📥 Export"])
            
            with tab1:
                st.header("🏆 Top Potency-Enhancing Transformations")
                
                # Sort by mean_delta descending for positive effects
                top_positive = mmp_df.sort_values('mean_delta', ascending=False).head(num_top_transforms)
                
                for i, (idx, row) in enumerate(top_positive.iterrows()):
                    col1, col2, col3 = st.columns([2, 2, 3])
                    
                    with col1:
                        st.subheader(f"Rank #{i+1}")
                        st.write(f"**Transform:** `{row['Transform']}`")
                        st.write(f"**Mean ΔpIC50:** {row['mean_delta']:.3f} ± {row['std_delta']:.3f}")
                        st.write(f"**Occurrences:** {row['Count']}")
                    
                    with col2:
                        # Display reaction
                        try:
                            img = rxn_to_image(row['Transform'], width=400, height=200)
                            st.image(img, caption="Reaction Transform")
                        except:
                            st.write("Could not display reaction image")
                    
                    with col3:
                        # Display stripplot
                        fig = create_stripplot(row['Deltas'], figsize=(4, 2))
                        st.pyplot(fig)
                        plt.close(fig)
                    
                    # Show example compounds for this transform
                    transform_str = row['Transform']
                    compound_pairs = delta_df[delta_df['Transform'] == transform_str].head(3)
                    
                    if not compound_pairs.empty:
                        st.markdown("**Example Compound Pairs:**")
                        examples_data = []
                        for _, pair in compound_pairs.iterrows():
                            examples_data.extend([
                                {'SMILES': pair['SMILES_1'], 'Name': pair['Name_1'], 'pIC50': pair['pIC50_1']},
                                {'SMILES': pair['SMILES_2'], 'Name': pair['Name_2'], 'pIC50': pair['pIC50_2']}
                            ])
                        
                        examples_df = pd.DataFrame(examples_data)
                        if len(examples_df) > 0:
                            fig = display_compound_grid(examples_df)
                            st.pyplot(fig)
                            plt.close(fig)
                    
                    st.markdown("---")
            
            with tab2:
                st.header("⚠️ Top Potency-Diminishing Transformations")
                
                # Sort by mean_delta ascending for negative effects
                top_negative = mmp_df.sort_values('mean_delta', ascending=True).head(num_top_transforms)
                
                for i, (idx, row) in enumerate(top_negative.iterrows()):
                    col1, col2, col3 = st.columns([2, 2, 3])
                    
                    with col1:
                        st.subheader(f"Rank #{i+1}")
                        st.write(f"**Transform:** `{row['Transform']}`")
                        st.write(f"**Mean ΔpIC50:** {row['mean_delta']:.3f} ± {row['std_delta']:.3f}")
                        st.write(f"**Occurrences:** {row['Count']}")
                    
                    with col2:
                        # Display reaction
                        try:
                            img = rxn_to_image(row['Transform'], width=400, height=200)
                            st.image(img, caption="Reaction Transform")
                        except:
                            st.write("Could not display reaction image")
                    
                    with col3:
                        # Display stripplot
                        fig = create_stripplot(row['Deltas'], figsize=(4, 2))
                        st.pyplot(fig)
                        plt.close(fig)
                    
                    # Show example compounds for this transform
                    transform_str = row['Transform']
                    compound_pairs = delta_df[delta_df['Transform'] == transform_str].head(3)
                    
                    if not compound_pairs.empty:
                        st.markdown("**Example Compound Pairs:**")
                        examples_data = []
                        for _, pair in compound_pairs.iterrows():
                            examples_data.extend([
                                {'SMILES': pair['SMILES_1'], 'Name': pair['Name_1'], 'pIC50': pair['pIC50_1']},
                                {'SMILES': pair['SMILES_2'], 'Name': pair['Name_2'], 'pIC50': pair['pIC50_2']}
                            ])
                        
                        examples_df = pd.DataFrame(examples_data)
                        if len(examples_df) > 0:
                            fig = display_compound_grid(examples_df)
                            st.pyplot(fig)
                            plt.close(fig)
                    
                    st.markdown("---")
            
            with tab3:
                st.header("📋 All Significant Transformations")
                
                # Sort all transforms by mean_delta
                mmp_df_sorted = mmp_df.sort_values('mean_delta', ascending=False)
                
                # Display as table
                display_df = mmp_df_sorted[['Transform', 'Count', 'mean_delta', 'std_delta']].copy()
                display_df['mean_delta'] = display_df['mean_delta'].round(3)
                display_df['std_delta'] = display_df['std_delta'].round(3)
                display_df.columns = ['Transform', 'Occurrences', 'Mean ΔpIC50', 'Std ΔpIC50']
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    column_config={
                        "Transform": st.column_config.TextColumn("Transform", width="large"),
                        "Occurrences": st.column_config.NumberColumn("Occurrences", width="small"),
                        "Mean ΔpIC50": st.column_config.NumberColumn("Mean ΔpIC50", width="small", format="%.3f"),
                        "Std ΔpIC50": st.column_config.NumberColumn("Std ΔpIC50", width="small", format="%.3f")
                    }
                )
                
                # Distribution plot of all mean deltas
                st.subheader("Distribution of Transformation Effects")
                fig, ax = plt.subplots(figsize=(10, 4))
                if len(mmp_df_sorted) > 1:
                    ax.hist(mmp_df_sorted['mean_delta'], bins=min(20, len(mmp_df_sorted)), alpha=0.7, color='steelblue', edgecolor='black')
                ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
                ax.set_xlabel('Mean ΔpIC50')
                ax.set_ylabel('Frequency')
                ax.set_title('Distribution of Transformation Effects')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close(fig)
            
            with tab4:
                st.header("💾 Export Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Export mmp_df
                    if len(mmp_df) > 0:
                        csv1 = mmp_df[['Transform', 'Count', 'mean_delta', 'std_delta']].to_csv(index=False)
                        st.download_button(
                            label="📥 Download Transformations (CSV)",
                            data=csv1,
                            file_name="mmp_transformations.csv",
                            mime="text/csv"
                        )
                    else:
                        st.write("No transformations to export")
                
                with col2:
                    # Export delta_df
                    if len(delta_df) > 0:
                        csv2 = delta_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download All Pairs (CSV)",
                            data=csv2,
                            file_name="mmp_pairs.csv",
                            mime="text/csv"
                        )
                    else:
                        st.write("No pairs to export")
                
                with col3:
                    # Export row_df (fragments)
                    if len(row_df) > 0:
                        csv3 = row_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Fragments (CSV)",
                            data=csv3,
                            file_name="mmp_fragments.csv",
                            mime="text/csv"
                        )
                    else:
                        st.write("No fragments to export")
                
                # Display summary
                st.subheader("Summary Statistics")
                summary_stats = {
                    'Total Compounds': len(df),
                    'Valid Compounds': len(df_processed) if df_processed is not None else 0,
                    'Fragment Pairs': len(row_df) if row_df is not None else 0,
                    'Molecular Pairs': len(delta_df) if delta_df is not None else 0,
                    'Significant Transforms': len(mmp_df) if mmp_df is not None else 0,
                    'Most Positive Transform': mmp_df.loc[mmp_df['mean_delta'].idxmax(), 'Transform'] if len(mmp_df) > 0 else 'N/A',
                    'Most Negative Transform': mmp_df.loc[mmp_df['mean_delta'].idxmin(), 'Transform'] if len(mmp_df) > 0 else 'N/A'
                }
                
                if len(mmp_df) > 0:
                    summary_stats['Avg Mean ΔpIC50'] = mmp_df['mean_delta'].mean()
                
                summary_df = pd.DataFrame(list(summary_stats.items()), columns=['Metric', 'Value'])
                st.dataframe(summary_df, use_container_width=True)
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

else:
    # Show instructions when no file is uploaded
    st.info("👈 Please upload a CSV file to begin analysis")
    
    # Show example data that works well
    st.header("📋 Example Data Format")
    st.markdown("""
    For best results, use data with compounds that:
    
    1. **Share common scaffolds** (same ring systems)
    2. **Have diverse substituents** (different R-groups)
    3. **Have measured pIC50 values** (potency data)
    
    **Example dataset that works well:**
    """)
    
    example_data = pd.DataFrame({
        'SMILES': [
            'Cc1ccc(cc1)C(=O)O',  # Benzoic acid with methyl
            'COc1ccc(cc1)C(=O)O', # Benzoic acid with methoxy
            'CFc1ccc(cc1)C(=O)O', # Benzoic acid with fluoro
            'CCc1ccc(cc1)C(=O)O', # Benzoic acid with ethyl
            'CCOc1ccc(cc1)C(=O)O', # Benzoic acid with ethoxy
            'CNc1ccc(cc1)C(=O)O', # Benzoic acid with amino
            'C(=O)Nc1ccc(cc1)C(=O)O', # Benzoic acid with amide
            'C(F)(F)Fc1ccc(cc1)C(=O)O', # Benzoic acid with trifluoromethyl
        ],
        'Name': ['Methyl_BA', 'Methoxy_BA', 'Fluoro_BA', 'Ethyl_BA', 
                'Ethoxy_BA', 'Amino_BA', 'Amide_BA', 'Trifluoro_BA'],
        'pIC50': [7.2, 7.8, 6.9, 7.5, 8.1, 6.5, 8.3, 5.8]
    })
    
    st.dataframe(example_data)
    
    # Create download link for example data
    csv = example_data.to_csv(index=False)
    st.download_button(
        label="📥 Download Example Data (CSV)",
        data=csv,
        file_name="example_mmp_data.csv",
        mime="text/csv",
        help="Download this example dataset to test the app"
    )
    
    # Installation instructions
    with st.expander("📦 Installation Instructions"):
        st.markdown("""
        ### For local installation:
        
        ```bash
        # Create a virtual environment (recommended)
        python -m venv mmp-env
        source mmp-env/bin/activate  # On Windows: mmp-env\\Scripts\\activate
        
        # Install dependencies
        pip install streamlit pandas matplotlib seaborn numpy pillow
        pip install rdkit-pypi
        ```
        
        ### For Streamlit Cloud:
        Add a `requirements.txt` file with:
        ```
        streamlit
        pandas
        matplotlib
        seaborn
        numpy
        pillow
        rdkit-pypi
        ```
        
        ### Run the app:
        ```bash
        streamlit run app.py
        ```
        """)

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
