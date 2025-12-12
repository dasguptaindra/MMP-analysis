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
Cc1ccc(cc1)C(=O)O,Compound1,7.2
COc1ccc(cc1)C(=O)O,Compound2,7.8
CFc1ccc(cc1)C(=O)O,Compound3,6.9
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

# Simplified fragmentation method using SMARTS patterns
def get_mmp_fragments_simple(mol):
    """Simple fragmentation method using common attachment points"""
    fragments = []
    
    try:
        # Convert to SMILES and look for common attachment patterns
        smiles = Chem.MolToSmiles(mol)
        
        # Try to fragment at common positions using SMARTS
        # Pattern for aromatic carbon with substituent
        aromatic_pattern = Chem.MolFromSmarts('[c;H1]')
        
        # Pattern for aliphatic carbon with substituent
        aliphatic_pattern = Chem.MolFromSmarts('[C;H2]')
        
        # Pattern for nitrogen with substituent
        nitrogen_pattern = Chem.MolFromSmarts('[N;H2]')
        
        patterns = [aromatic_pattern, aliphatic_pattern, nitrogen_pattern]
        pattern_names = ['aromatic', 'aliphatic', 'nitrogen']
        
        for pattern, name in zip(patterns, pattern_names):
            if pattern:
                matches = mol.GetSubstructMatches(pattern)
                for match in matches:
                    if len(match) > 0:
                        atom_idx = match[0]
                        
                        # Create a modified molecule with wildcard
                        mol_copy = Chem.RWMol(mol)
                        
                        # Get the atom
                        atom = mol_copy.GetAtomWithIdx(atom_idx)
                        
                        # Create scaffold: replace the atom with wildcard
                        wildcard = Chem.Atom(0)  # Wildcard atom
                        wildcard.SetAtomMapNum(1)
                        mol_copy.ReplaceAtom(atom_idx, wildcard)
                        
                        # Create substituent: extract the atom and its neighbors
                        substituent_mol = Chem.RWMol()
                        
                        # Add the atom with wildcard
                        sub_atom = Chem.Atom(atom.GetAtomicNum())
                        sub_atom.SetAtomMapNum(2)
                        sub_idx = substituent_mol.AddAtom(sub_atom)
                        
                        # Add the atom's neighbors (except hydrogens)
                        atom_map = {atom_idx: sub_idx}
                        for neighbor in atom.GetNeighbors():
                            if neighbor.GetAtomicNum() != 1:  # Skip hydrogens
                                new_atom = Chem.Atom(neighbor.GetAtomicNum())
                                new_idx = substituent_mol.AddAtom(new_atom)
                                atom_map[neighbor.GetIdx()] = new_idx
                                
                                # Add bond
                                bond_type = mol.GetBondBetweenAtoms(atom_idx, neighbor.GetIdx()).GetBondType()
                                substituent_mol.AddBond(sub_idx, new_idx, bond_type)
                        
                        # Clean up molecules
                        try:
                            scaffold = mol_copy.GetMol()
                            substituent = substituent_mol.GetMol()
                            
                            Chem.SanitizeMol(scaffold)
                            Chem.SanitizeMol(substituent)
                            
                            fragments.append((scaffold, substituent))
                        except:
                            continue
        
        # If no fragments found with patterns, try a simpler approach
        if not fragments:
            # Just split at the first non-ring single bond
            for bond in mol.GetBonds():
                if bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
                    a1 = bond.GetBeginAtom()
                    a2 = bond.GetEndAtom()
                    
                    # Create fragments by breaking this bond
                    mol_copy = Chem.RWMol(mol)
                    
                    # Replace both atoms with wildcards
                    a1_idx = bond.GetBeginAtomIdx()
                    a2_idx = bond.GetEndAtomIdx()
                    
                    wildcard1 = Chem.Atom(0)
                    wildcard1.SetAtomMapNum(1)
                    mol_copy.ReplaceAtom(a1_idx, wildcard1)
                    
                    wildcard2 = Chem.Atom(0)
                    wildcard2.SetAtomMapNum(2)
                    mol_copy.ReplaceAtom(a2_idx, wildcard2)
                    
                    try:
                        scaffold = mol_copy.GetMol()
                        Chem.SanitizeMol(scaffold)
                        
                        # Create substituent as just a wildcard
                        substituent = Chem.MolFromSmiles('*')
                        substituent.GetAtomWithIdx(0).SetAtomMapNum(2)
                        
                        fragments.append((scaffold, substituent))
                    except:
                        continue
        
        # Remove duplicates
        unique_fragments = []
        seen = set()
        for scaffold, substituent in fragments:
            scaffold_smiles = Chem.MolToSmiles(scaffold)
            substituent_smiles = Chem.MolToSmiles(substituent)
            key = (scaffold_smiles, substituent_smiles)
            if key not in seen:
                seen.add(key)
                unique_fragments.append((scaffold, substituent))
        
        return unique_fragments
    
    except Exception as e:
        st.warning(f"Fragmentation error: {e}")
        return []

# Even simpler fragmentation - just use Murcko scaffolds
def get_murcko_fragments(mol):
    """Use RDKit's Murcko scaffold decomposition"""
    from rdkit.Chem.Scaffolds import MurckoScaffold
    
    try:
        # Get Murcko scaffold
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        
        # Get sidechains by removing scaffold from molecule
        scaffold_atoms = set(scaffold.GetAtoms())
        mol_atoms = set(mol.GetAtoms())
        
        # Find atoms that are not in scaffold
        sidechain_atoms = mol_atoms - scaffold_atoms
        
        if len(sidechain_atoms) > 0:
            # Create sidechain molecule
            sidechain_mol = Chem.RWMol()
            atom_map = {}
            
            # Add sidechain atoms
            for atom in sidechain_atoms:
                new_atom = Chem.Atom(atom.GetAtomicNum())
                new_idx = sidechain_mol.AddAtom(new_atom)
                atom_map[atom.GetIdx()] = new_idx
            
            # Add bonds between sidechain atoms
            for atom in sidechain_atoms:
                for neighbor in atom.GetNeighbors():
                    if neighbor in sidechain_atoms:
                        bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
                        if bond and atom.GetIdx() < neighbor.GetIdx():
                            sidechain_mol.AddBond(
                                atom_map[atom.GetIdx()],
                                atom_map[neighbor.GetIdx()],
                                bond.GetBondType()
                            )
            
            # Add attachment point wildcards
            scaffold_with_wildcard = Chem.RWMol(scaffold)
            for atom in scaffold.GetAtoms():
                for neighbor in atom.GetNeighbors():
                    if neighbor in sidechain_atoms:
                        # Replace with wildcard in scaffold
                        wildcard = Chem.Atom(0)
                        wildcard.SetAtomMapNum(1)
                        scaffold_with_wildcard.ReplaceAtom(atom.GetIdx(), wildcard)
                        
                        # Add wildcard to sidechain
                        sidechain_atom = sidechain_mol.GetAtomWithIdx(atom_map[neighbor.GetIdx()])
                        sidechain_atom.SetAtomMapNum(2)
            
            try:
                scaffold_final = scaffold_with_wildcard.GetMol()
                sidechain_final = sidechain_mol.GetMol()
                Chem.SanitizeMol(scaffold_final)
                Chem.SanitizeMol(sidechain_final)
                return [(scaffold_final, sidechain_final)]
            except:
                return []
        
        return []
    
    except Exception as e:
        st.warning(f"Murcko fragmentation error: {e}")
        return []

# Ultra-simple fragmentation - just extract common substituents
def get_simple_substituents(mol):
    """Extract common substituents from molecules"""
    fragments = []
    
    try:
        # Common substituent patterns
        substituents = [
            ('[CH3]', 'Methyl'),
            ('[OH]', 'Hydroxy'),
            ('[NH2]', 'Amino'),
            ('[F]', 'Fluoro'),
            ('[Cl]', 'Chloro'),
            ('[Br]', 'Bromo'),
            ('[I]', 'Iodo'),
            ('[OCH3]', 'Methoxy'),
            ('[N+]', 'Ammonium'),
            ('[C=O]', 'Carbonyl'),
        ]
        
        for sub_smarts, name in substituents:
            pattern = Chem.MolFromSmarts(sub_smarts)
            if pattern:
                matches = mol.GetSubstructMatches(pattern)
                for match in matches:
                    if len(match) > 0:
                        # Create scaffold by removing this substituent
                        scaffold_mol = Chem.RWMol(mol)
                        
                        # Remove the matched atoms
                        for atom_idx in sorted(match, reverse=True):
                            try:
                                scaffold_mol.RemoveAtom(atom_idx)
                            except:
                                pass
                        
                        # Add wildcard at attachment point
                        if scaffold_mol.GetNumAtoms() > 0:
                            # Create substituent as the pattern with wildcard
                            substituent_mol = Chem.MolFromSmiles(sub_smarts.replace('[', '').replace(']', ''))
                            if substituent_mol:
                                # Add wildcard
                                for atom in substituent_mol.GetAtoms():
                                    if atom.GetDegree() == 1:  # Terminal atom
                                        atom.SetAtomMapNum(2)
                                        break
                                
                                # Add wildcard to scaffold
                                for atom in scaffold_mol.GetAtoms():
                                    if atom.GetDegree() < atom.GetExplicitValence():
                                        atom.SetAtomMapNum(1)
                                        break
                                
                                try:
                                    scaffold = scaffold_mol.GetMol()
                                    substituent = substituent_mol
                                    Chem.SanitizeMol(scaffold)
                                    Chem.SanitizeMol(substituent)
                                    fragments.append((scaffold, substituent))
                                except:
                                    continue
        
        return fragments
    
    except Exception as e:
        st.warning(f"Simple substituent error: {e}")
        return []

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
            # Try multiple fragmentation methods
            fragments = []
            
            # Method 1: Simple SMARTS-based
            fragments1 = get_mmp_fragments_simple(mol)
            if fragments1:
                fragments.extend(fragments1)
            
            # Method 2: Murcko scaffolds
            fragments2 = get_murcko_fragments(mol)
            if fragments2:
                fragments.extend(fragments2)
            
            # Method 3: Simple substituents
            fragments3 = get_simple_substituents(mol)
            if fragments3:
                fragments.extend(fragments3)
            
            # Remove duplicates
            unique_fragments = []
            seen = set()
            for scaffold_mol, substituent_mol in fragments:
                try:
                    scaffold_smiles = Chem.MolToSmiles(scaffold_mol)
                    substituent_smiles = Chem.MolToSmiles(substituent_mol)
                    key = (scaffold_smiles, substituent_smiles)
                    if key not in seen:
                        seen.add(key)
                        unique_fragments.append((scaffold_mol, substituent_mol))
                except:
                    continue
            
            for scaffold_mol, substituent_mol in unique_fragments:
                try:
                    scaffold_smiles = Chem.MolToSmiles(scaffold_mol)
                    substituent_smiles = Chem.MolToSmiles(substituent_mol)
                    
                    # Add to results
                    row_list.append([smiles, scaffold_smiles, substituent_smiles, name, pIC50])
                except:
                    continue
        
        except Exception as e:
            st.warning(f"Error processing molecule {name}: {e}")
        
        progress_bar.progress((idx + 1) / len(valid_df))
    
    if not row_list:
        st.error("No valid fragments generated.")
        st.error("This could be because:")
        st.error("1. Molecules are too simple or don't have clear substituents")
        st.error("2. Try using compounds with common scaffolds and different R-groups")
        return None, None, None, None
    
    row_df = pd.DataFrame(row_list, columns=["SMILES", "Core", "R_group", "Name", "pIC50"])
    st.info(f"✅ Generated {len(row_df)} fragment pairs.")
    
    # Collect pairs with same scaffold
    st.info("🤝 Finding molecular pairs...")
    delta_list = []
    
    # Group by scaffold and find pairs
    for core_smiles, group in row_df.groupby("Core"):
        if len(group) > 1:
            compounds = group.to_dict('records')
            
            # Create all possible pairs
            for i in range(len(compounds)):
                for j in range(i+1, len(compounds)):
                    comp_a = compounds[i]
                    comp_b = compounds[j]
                    
                    if comp_a['SMILES'] != comp_b['SMILES']:
                        # Calculate delta pIC50
                        delta = comp_b['pIC50'] - comp_a['pIC50']
                        
                        # Create transform string
                        transform = f"{comp_a['R_group']}>>{comp_b['R_group']}"
                        
                        delta_list.append([
                            comp_a['SMILES'], comp_a['Core'], comp_a['R_group'], comp_a['Name'], comp_a['pIC50'],
                            comp_b['SMILES'], comp_b['Core'], comp_b['R_group'], comp_b['Name'], comp_b['pIC50'],
                            transform, delta
                        ])
    
    if not delta_list:
        st.error("No molecular pairs found.")
        return None, None, None, None
    
    cols = ["SMILES_1", "Core_1", "R_group_1", "Name_1", "pIC50_1",
           "SMILES_2", "Core_2", "Rgroup_1", "Name_2", "pIC50_2",
           "Transform", "Delta"]
    delta_df = pd.DataFrame(delta_list, columns=cols)
    st.info(f"✅ Found {len(delta_df)} molecular pairs.")
    
    # Collect frequently occurring pairs
    st.info("📊 Analyzing transformations...")
    mmp_list = []
    transform_to_idx = {}
    
    for idx, (transform, group) in enumerate(delta_df.groupby("Transform")):
        if len(group) >= min_occurrences:
            mmp_list.append([transform, len(group), group.Delta.values])
            transform_to_idx[transform] = idx
    
    if not mmp_list:
        st.warning(f"No transformations found with at least {min_occurrences} occurrences.")
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
            st.info("Suggestions:")
            st.info("1. Lower the minimum occurrence threshold (try 1)")
            st.info("2. Use compounds with a common scaffold and different R-groups")
            st.info("3. Try the example data below to see how it should work")
            
            # Show fragment preview if available
            if row_df is not None and len(row_df) > 0:
                st.subheader("Generated Fragments Preview")
                st.dataframe(row_df.head(), use_container_width=True)
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
                }
                
                if len(mmp_df) > 0:
                    summary_stats['Avg Mean ΔpIC50'] = f"{mmp_df['mean_delta'].mean():.3f}"
                    summary_stats['Most Positive Transform'] = mmp_df.loc[mmp_df['mean_delta'].idxmax(), 'Transform']
                    summary_stats['Most Negative Transform'] = mmp_df.loc[mmp_df['mean_delta'].idxmin(), 'Transform']
                
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
    **For best results, use data like this:**
    
    Compounds that share a common scaffold (like benzoic acid) with different R-groups.
    """)
    
    example_data = pd.DataFrame({
        'SMILES': [
            'Cc1ccccc1C(=O)O',      # Benzoic acid with methyl at position 1
            'COc1ccccc1C(=O)O',     # Benzoic acid with methoxy at position 1
            'FC1=C(C(=O)O)C=CC=C1', # Benzoic acid with fluoro at position 2
            'Clc1ccccc1C(=O)O',     # Benzoic acid with chloro at position 1
            'BrC1=CC=CC=C1C(=O)O',  # Benzoic acid with bromo at position 1
            'IC1=CC=CC=C1C(=O)O',   # Benzoic acid with iodo at position 1
            'OC1=CC=CC=C1C(=O)O',   # Benzoic acid with hydroxy at position 1
            'NC1=CC=CC=C1C(=O)O',   # Benzoic acid with amino at position 1
        ],
        'Name': ['Methyl_BA', 'Methoxy_BA', 'Fluoro_BA', 'Chloro_BA', 
                'Bromo_BA', 'Iodo_BA', 'Hydroxy_BA', 'Amino_BA'],
        'pIC50': [7.2, 7.8, 6.9, 7.5, 7.1, 6.8, 8.0, 6.5]
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
    
    # Test with example data
    st.header("🚀 Test with Example Data")
    if st.button("Run Analysis with Example Data"):
        with st.spinner("Running analysis with example data..."):
            try:
                result = run_mmp_analysis(example_data, 1)  # Use min occurrence of 1 for demo
                
                if result[0] is not None:
                    df_processed, row_df, delta_df, mmp_df = result
                    
                    if len(mmp_df) > 0:
                        st.success(f"✅ Found {len(mmp_df)} transformations!")
                        
                        # Show top transform
                        st.subheader("Top Transformation")
                        top_transform = mmp_df.sort_values('mean_delta', ascending=False).iloc[0]
                        st.write(f"**Transform:** `{top_transform['Transform']}`")
                        st.write(f"**Mean ΔpIC50:** {top_transform['mean_delta']:.3f}")
                        st.write(f"**Occurrences:** {top_transform['Count']}")
                    else:
                        st.warning("No transformations found with example data.")
                        if row_df is not None:
                            st.subheader("Generated Fragments")
                            st.dataframe(row_df.head())
                else:
                    st.error("Failed to run analysis with example data.")
            except Exception as e:
                st.error(f"Error: {e}")

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
