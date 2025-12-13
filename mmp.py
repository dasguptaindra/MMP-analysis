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
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Scaffolds import MurckoScaffold
    rdDepictor.SetPreferCoordGen(True)
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

# Visualization parameters
st.sidebar.header("🎨 Visualization Settings")
show_smiles = st.sidebar.checkbox("Show SMILES", value=True, help="Display SMILES strings")
show_molecules = st.sidebar.checkbox("Show molecule images", value=True, help="Display molecule structures")
highlight_common_core = st.sidebar.checkbox("Highlight common core", value=True, help="Color-code the common scaffold")
image_size = st.sidebar.slider("Molecule image size", 200, 400, 300, help="Size of molecule images")

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

# NEW: Function to highlight common cores in molecules
def highlight_common_scaffold(mol1, mol2, highlight_color=(0.8, 0.8, 0.8, 0.6)):
    """Highlight the common scaffold between two molecules"""
    try:
        # Find maximum common substructure
        from rdkit.Chem import rdFMCS
        mcs_result = rdFMCS.FindMCS([mol1, mol2], timeout=60)
        
        if mcs_result.numAtoms > 0:
            mcs_smarts = mcs_result.smartsString
            pattern = Chem.MolFromSmarts(mcs_smarts)
            
            if pattern:
                # Highlight matching atoms in mol1
                matches1 = mol1.GetSubstructMatches(pattern)
                matches2 = mol2.GetSubstructMatches(pattern)
                
                if matches1 and matches2:
                    highlight_atoms1 = list(matches1[0])
                    highlight_atoms2 = list(matches2[0])
                    
                    return highlight_atoms1, highlight_atoms2
    except:
        pass
    return [], []

# NEW: Improved molecule visualization with highlighting
def draw_molecule_pair(mol1, mol2, name1="Compound 1", name2="Compound 2", 
                      pIC50_1=None, pIC50_2=None, highlight_core=True, size=(300, 300)):
    """Draw two molecules side by side with highlighting"""
    from rdkit.Chem.Draw import MolsToGridImage
    
    # Prepare molecules for display
    mols = [mol1, mol2]
    
    # Highlight common core if requested
    highlight_atoms = []
    highlight_bonds = []
    
    if highlight_core and mol1 and mol2:
        atoms1, atoms2 = highlight_common_scaffold(mol1, mol2)
        
        # Create highlight colors
        colors1 = [(0.8, 0.8, 0.8, 0.6)] * len(atoms1) if atoms1 else []
        colors2 = [(0.8, 0.8, 0.8, 0.6)] * len(atoms2) if atoms2 else []
        
        highlight_atoms = [atoms1, atoms2]
    
    # Create labels
    labels = []
    for i, (mol, pIC50) in enumerate(zip([mol1, mol2], [pIC50_1, pIC50_2])):
        label_parts = [f"Compound {i+1}"]
        if pIC50 is not None:
            label_parts.append(f"pIC50: {pIC50:.2f}")
        labels.append("\n".join(label_parts))
    
    # Draw molecules
    try:
        img = MolsToGridImage(mols, molsPerRow=2, subImgSize=size,
                             legends=labels,
                             highlightAtomLists=highlight_atoms if highlight_atoms else None,
                             highlightBondLists=highlight_bonds if highlight_bonds else None)
        return img
    except:
        # Fallback to simple drawing
        img = MolsToGridImage(mols, molsPerRow=2, subImgSize=size, legends=labels)
        return img

# NEW: Enhanced transformation visualization
def visualize_transformation_enhanced(transform_str, compounds_df=None, 
                                     example_compounds=None, size=(400, 200)):
    """Create an enhanced visualization of the transformation"""
    try:
        if '>>' in transform_str:
            reactant_str, product_str = transform_str.split('>>')
            
            # Create a reaction visualization
            rxn_smarts = f"[*:1]{reactant_str}>>[*:1]{product_str}"
            
            try:
                rxn = AllChem.ReactionFromSmarts(rxn_smarts)
                
                # Create a more detailed image
                drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
                drawer.DrawReaction(rxn)
                drawer.FinishDrawing()
                
                # Get PNG data
                png_data = drawer.GetDrawingText()
                
                # Convert to PIL Image
                img = Image.open(io.BytesIO(png_data))
                return img
                
            except:
                # Fallback: Show reactants and products separately
                from rdkit.Chem.Draw import MolsToGridImage
                
                # Create placeholder molecules
                reactant_mol = Chem.MolFromSmiles(f"[*]{reactant_str}")
                product_mol = Chem.MolFromSmiles(f"[*]{product_str}")
                
                if reactant_mol and product_mol:
                    mols = [reactant_mol, product_mol]
                    img = MolsToGridImage(mols, molsPerRow=2, 
                                         subImgSize=(size[0]//2, size[1]),
                                         legends=["Reactant", "Product"])
                    return img
                else:
                    # Create text-based visualization
                    return create_text_visualization(transform_str, size)
        else:
            return create_text_visualization(transform_str, size)
            
    except Exception as e:
        return create_text_visualization(f"Error: {str(e)[:50]}", size)

def create_text_visualization(text, size):
    """Create a text-based visualization as fallback"""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to load font
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()
    
    # Split text if too long
    if len(text) > 40:
        lines = [text[i:i+40] for i in range(0, len(text), 40)]
    else:
        lines = [text]
    
    # Draw lines
    y_offset = 10
    for line in lines:
        draw.text((10, y_offset), line, fill='black', font=font)
        y_offset += 20
    
    return img

# ORIGINAL FRAGMENTATION FUNCTIONS (Must be included)
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

def get_murcko_fragments(mol):
    """Use RDKit's Murcko scaffold decomposition"""
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

# NEW: Function to display transformation with examples
def display_transformation_with_examples(transform_str, delta_df, compounds_df, 
                                        transform_idx, highlight_core=True):
    """Display a transformation with example compound pairs"""
    
    # Get all pairs with this transformation
    transform_pairs = delta_df[delta_df['Transform'] == transform_str]
    
    if len(transform_pairs) == 0:
        return None
    
    # Get the first few example pairs
    example_pairs = transform_pairs.head(3).to_dict('records')
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Summary", "🧪 Example Pairs", "📈 Statistics"])
    
    with tab1:
        # Display transformation summary
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Display the transformation
            st.subheader("Transformation")
            
            # Show SMILES if enabled
            if show_smiles:
                st.code(transform_str, language="text")
            
            # Show molecule visualization if enabled
            if show_molecules:
                img = visualize_transformation_enhanced(transform_str, size=(image_size, image_size//2))
                st.image(img, caption="Structural Transformation")
        
        with col2:
            # Display statistics
            st.subheader("Statistics")
            
            stats_df = pd.DataFrame({
                'Metric': ['Mean ΔpIC50', 'Standard Deviation', 'Number of Occurrences', 
                          'Min ΔpIC50', 'Max ΔpIC50'],
                'Value': [f"{transform_pairs['Delta'].mean():.3f}", 
                         f"{transform_pairs['Delta'].std():.3f}",
                         f"{len(transform_pairs)}",
                         f"{transform_pairs['Delta'].min():.3f}",
                         f"{transform_pairs['Delta'].max():.3f}"]
            })
            
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
            
            # Delta distribution
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.hist(transform_pairs['Delta'].values, bins=10, alpha=0.7, color='steelblue')
            ax.axvline(transform_pairs['Delta'].mean(), color='red', linestyle='--', 
                      label=f'Mean: {transform_pairs["Delta"].mean():.3f}')
            ax.set_xlabel('ΔpIC50')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of ΔpIC50 Values')
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)
    
    with tab2:
        # Display example compound pairs
        st.subheader("Example Compound Pairs")
        
        for i, pair in enumerate(example_pairs):
            with st.expander(f"Example Pair {i+1}", expanded=(i==0)):
                col1, col2, col3 = st.columns([2, 1, 2])
                
                # Get molecule objects
                mol1 = Chem.MolFromSmiles(pair['SMILES_1'])
                mol2 = Chem.MolFromSmiles(pair['SMILES_2'])
                
                with col1:
                    st.markdown(f"**{pair['Name_1']}**")
                    if show_smiles:
                        st.code(pair['SMILES_1'], language="text")
                    if show_molecules and mol1:
                        img = Draw.MolToImage(mol1, size=(200, 200))
                        st.image(img, caption=f"pIC50: {pair['pIC50_1']:.2f}")
                
                with col2:
                    st.markdown("### →")
                    delta = pair['Delta']
                    color = "green" if delta > 0 else "red" if delta < 0 else "gray"
                    st.markdown(f"<h3 style='color:{color}; text-align:center'>Δ = {delta:.3f}</h3>", 
                               unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"**{pair['Name_2']}**")
                    if show_smiles:
                        st.code(pair['SMILES_2'], language="text")
                    if show_molecules and mol2:
                        img = Draw.MolToImage(mol2, size=(200, 200))
                        st.image(img, caption=f"pIC50: {pair['pIC50_2']:.2f}")
                
                # Display side-by-side comparison if both molecules exist
                if mol1 and mol2 and show_molecules:
                    st.markdown("**Side-by-side comparison:**")
                    try:
                        comparison_img = draw_molecule_pair(
                            mol1, mol2, 
                            name1=pair['Name_1'], 
                            name2=pair['Name_2'],
                            pIC50_1=pair['pIC50_1'],
                            pIC50_2=pair['pIC50_2'],
                            highlight_core=highlight_common_core,
                            size=(250, 250)
                        )
                        st.image(comparison_img)
                    except:
                        pass
    
    with tab3:
        # Detailed statistics
        st.subheader("Detailed Statistics")
        
        # Create a dataframe with all pairs
        detailed_df = transform_pairs[['Name_1', 'pIC50_1', 'Name_2', 'pIC50_2', 'Delta']].copy()
        detailed_df['Delta'] = detailed_df['Delta'].round(3)
        
        st.dataframe(detailed_df, use_container_width=True)
        
        # Additional statistics
        col1, col2 = st.columns(2)
        
        with col1:
            # Box plot
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.boxplot(transform_pairs['Delta'].values, vert=False)
            ax.set_xlabel('ΔpIC50')
            ax.set_title('Box Plot of ΔpIC50')
            st.pyplot(fig)
            plt.close(fig)
        
        with col2:
            # QQ plot for normality check
            try:
                from scipy import stats
                fig, ax = plt.subplots(figsize=(4, 3))
                stats.probplot(transform_pairs['Delta'].values, dist="norm", plot=ax)
                ax.set_title('Q-Q Plot')
                st.pyplot(fig)
                plt.close(fig)
            except:
                st.info("Install scipy for Q-Q plots: pip install scipy")
    
    return example_pairs

# Modified rxn_to_image function to use enhanced visualization
def rxn_to_image(rxn_smarts, width=300, height=150):
    """Convert reaction SMARTS to PIL Image"""
    try:
        return visualize_transformation_enhanced(rxn_smarts, size=(width, height))
    except:
        # Fallback
        from PIL import Image, ImageDraw
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((10, height//2 - 10), str(rxn_smarts)[:50], fill='black')
        return img

# Plotting functions
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

# Modified analysis function to use enhanced display
def run_mmp_analysis(df, min_occurrences=2):
    """Main MMP analysis pipeline"""
    
    # Process molecules
    df['mol'] = df['SMILES'].apply(Chem.MolFromSmiles)
    valid_df = df[df['mol'].notna()].copy()
    
    if len(valid_df) == 0:
        st.error("No valid molecules found in the dataset.")
        return None, None, None, None
    
    st.info(f"✅ Found {len(valid_df)} valid molecules out of {len(df)} total.")
    
    # Display all compounds
    if show_molecules:
        with st.expander("📋 View All Compounds", expanded=False):
            st.subheader("All Compounds in Dataset")
            
            # Create tabs for different views
            tab_view1, tab_view2 = st.tabs(["Grid View", "Table View"])
            
            with tab_view1:
                # Display as grid of molecules
                n_cols = 4
                for i in range(0, len(valid_df), n_cols):
                    cols = st.columns(n_cols)
                    for j in range(n_cols):
                        idx = i + j
                        if idx < len(valid_df):
                            with cols[j]:
                                row = valid_df.iloc[idx]
                                mol = row['mol']
                                if mol:
                                    img = Draw.MolToImage(mol, size=(200, 200))
                                    st.image(img, caption=f"{row['Name']}\npIC50: {row['pIC50']:.2f}")
            
            with tab_view2:
                # Display as table with SMILES
                display_df = valid_df[['Name', 'SMILES', 'pIC50']].copy()
                st.dataframe(display_df, use_container_width=True)
    
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
        
        # Display data preview with enhanced view
        st.header("📊 Data Preview")
        
        col_preview1, col_preview2 = st.columns([1, 2])
        
        with col_preview1:
            st.dataframe(df.head(), use_container_width=True)
            st.write(f"Total compounds: {len(df)}")
        
        with col_preview2:
            # Basic statistics in a nicer format
            st.subheader("📈 Basic Statistics")
            stats_cols = st.columns(4)
            stats_data = {
                "Mean pIC50": f"{df['pIC50'].mean():.2f}",
                "Min pIC50": f"{df['pIC50'].min():.2f}",
                "Max pIC50": f"{df['pIC50'].max():.2f}",
                "Std Dev": f"{df['pIC50'].std():.2f}"
            }
            
            for (label, value), col in zip(stats_data.items(), stats_cols):
                with col:
                    st.metric(label, value)
            
            # Distribution plot
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.hist(df['pIC50'].values, bins=15, alpha=0.7, color='steelblue', edgecolor='black')
            ax.set_xlabel('pIC50')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of pIC50 Values')
            st.pyplot(fig)
            plt.close(fig)
        
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
                    st.markdown(f"## Transformation #{i+1}")
                    
                    # Use the enhanced display function
                    display_transformation_with_examples(
                        row['Transform'], 
                        delta_df, 
                        df_processed,
                        i,
                        highlight_common_core
                    )
                    
                    st.markdown("---")
            
            with tab2:
                st.header("⚠️ Top Potency-Diminishing Transformations")
                
                # Sort by mean_delta ascending for negative effects
                top_negative = mmp_df.sort_values('mean_delta', ascending=True).head(num_top_transforms)
                
                for i, (idx, row) in enumerate(top_negative.iterrows()):
                    st.markdown(f"## Transformation #{i+1}")
                    
                    # Use the enhanced display function
                    display_transformation_with_examples(
                        row['Transform'], 
                        delta_df, 
                        df_processed,
                        i,
                        highlight_common_core
                    )
                    
                    st.markdown("---")
            
            with tab3:
                st.header("📋 All Significant Transformations")
                
                # Sort all transforms by mean_delta
                mmp_df_sorted = mmp_df.sort_values('mean_delta', ascending=False)
                
                # Create an interactive table
                display_df = mmp_df_sorted[['Transform', 'Count', 'mean_delta', 'std_delta']].copy()
                display_df['mean_delta'] = display_df['mean_delta'].round(3)
                display_df['std_delta'] = display_df['std_delta'].round(3)
                display_df.columns = ['Transform', 'Occurrences', 'Mean ΔpIC50', 'Std ΔpIC50']
                
                # Add color coding for ΔpIC50
                def color_negative_red(val):
                    color = 'red' if val < 0 else ('green' if val > 0 else 'black')
                    return f'color: {color}'
                
                styled_df = display_df.style.applymap(color_negative_red, subset=['Mean ΔpIC50'])
                
                st.dataframe(
                    styled_df,
                    use_container_width=True,
                    column_config={
                        "Transform": st.column_config.TextColumn(
                            "Transform", 
                            width="large",
                            help="Structural transformation pattern"
                        ),
                        "Occurrences": st.column_config.NumberColumn(
                            "Occurrences", 
                            width="small",
                            help="Number of times this transformation was observed"
                        ),
                        "Mean ΔpIC50": st.column_config.NumberColumn(
                            "Mean ΔpIC50", 
                            width="small", 
                            format="%.3f",
                            help="Average change in potency"
                        ),
                        "Std ΔpIC50": st.column_config.NumberColumn(
                            "Std ΔpIC50", 
                            width="small", 
                            format="%.3f",
                            help="Standard deviation of potency change"
                        )
                    }
                )
                
                # Allow users to click on a transformation for details
                st.subheader("🔍 View Transformation Details")
                selected_transform = st.selectbox(
                    "Select a transformation to view details:",
                    options=mmp_df_sorted['Transform'].tolist(),
                    format_func=lambda x: f"{x[:50]}..." if len(x) > 50 else x
                )
                
                if selected_transform:
                    selected_row = mmp_df_sorted[mmp_df_sorted['Transform'] == selected_transform].iloc[0]
                    display_transformation_with_examples(
                        selected_row['Transform'], 
                        delta_df, 
                        df_processed,
                        0,
                        highlight_common_core
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
                    
                    # Get top and bottom transforms
                    top_transform = mmp_df.loc[mmp_df['mean_delta'].idxmax()]
                    bottom_transform = mmp_df.loc[mmp_df['mean_delta'].idxmin()]
                    
                    summary_stats['Most Positive Transform'] = f"{top_transform['Transform'][:50]}... (Δ={top_transform['mean_delta']:.3f})"
                    summary_stats['Most Negative Transform'] = f"{bottom_transform['Transform'][:50]}... (Δ={bottom_transform['mean_delta']:.3f})"
                
                summary_df = pd.DataFrame(list(summary_stats.items()), columns=['Metric', 'Value'])
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
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
    
    # Display example molecules
    if show_molecules:
        st.subheader("Example Molecules")
        from rdkit.Chem.Draw import MolsToGridImage
        
        mols = []
        legends = []
        for _, row in example_data.iterrows():
            mol = Chem.MolFromSmiles(row['SMILES'])
            if mol:
                mols.append(mol)
                legends.append(f"{row['Name']}\npIC50: {row['pIC50']}")
        
        if mols:
            img = MolsToGridImage(mols, molsPerRow=4, subImgSize=(200, 200), legends=legends)
            st.image(img, caption="Example Compounds")
    
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
                        
                        # Show top transform using enhanced display
                        st.subheader("Top Transformation")
                        top_transform = mmp_df.sort_values('mean_delta', ascending=False).iloc[0]
                        
                        display_transformation_with_examples(
                            top_transform['Transform'], 
                            delta_df, 
                            df_processed,
                            0,
                            highlight_common_core
                        )
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
