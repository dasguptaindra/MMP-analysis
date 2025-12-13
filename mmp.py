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
import itertools

# Try to import RDKit with error handling
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw, rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D, MolsToGridImage
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.Chem import rdFMCS
    from rdkit.Chem.rdMMPA import FragmentMol
    from rdkit.Chem.rdRGroupDecomposition import RGroupDecompose
    rdDepictor.SetPreferCoordGen(True)
    RDKIT_AVAILABLE = True
except ImportError as e:
    st.error(f"RDKit import error: {e}")
    RDKIT_AVAILABLE = False

# Set page config
st.set_page_config(
    page_title="MMP Analysis & Scaffold Finder",
    page_icon="🧪",
    layout="wide"
)

# Title
st.title("🧪 Matched Molecular Pair (MMP) Analysis & Scaffold Finder")
st.markdown("""
This app performs Matched Molecular Pair analysis to identify structural transformations 
that influence compound potency (pIC50 values) and finds common scaffolds in your dataset.
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

# ============================================================================
# FUNCTION DEFINITIONS (UPDATED WITH SCAFFOLD FINDER)
# ============================================================================

def get_largest_fragment(mol):
    """Get the largest fragment from a molecule"""
    try:
        frags = Chem.GetMolFrags(mol, asMols=True)
        if frags:
            return max(frags, key=lambda x: x.GetNumAtoms())
        return mol
    except:
        return mol

def cleanup_fragment(mol):
    """
    Replace atom map numbers with Hydrogens
    :param mol: input molecule
    :return: modified molecule, number of R-groups
    """
    rgroup_count = 0
    for atm in mol.GetAtoms():
        atm.SetAtomMapNum(0)
        if atm.GetAtomicNum() == 0:
            rgroup_count += 1
            atm.SetAtomicNum(1)
    mol = Chem.RemoveAllHs(mol)
    return mol, rgroup_count

def generate_fragments(mol, min_fraction=0.67):
    """
    Generate fragments using the RDKit's FragmentMol function
    :param mol: RDKit molecule
    :param min_fraction: minimum fraction of atoms to keep (0-1)
    :return: a Pandas dataframe with Scaffold SMILES, Number of Atoms, Number of R-Groups
    """
    # Generate molecule fragments
    frag_list = FragmentMol(mol)
    # Flatten the output into a single list
    flat_frag_list = [x for x in itertools.chain(*frag_list) if x]
    # The output of Fragment mol is contained in single molecules.  
    # Extract the largest fragment from each molecule
    flat_frag_list = [get_largest_fragment(x) for x in flat_frag_list]
    # Keep fragments where the number of atoms in the fragment is at least 
    # min_fraction of the number of atoms in input molecule
    num_mol_atoms = mol.GetNumAtoms()
    flat_frag_list = [x for x in flat_frag_list if x.GetNumAtoms() / num_mol_atoms > min_fraction]
    # Remove atom map numbers from the fragments
    flat_frag_list = [cleanup_fragment(x) for x in flat_frag_list]
    # Convert fragments to SMILES
    frag_smiles_list = [[Chem.MolToSmiles(x), x.GetNumAtoms(), y] for (x, y) in flat_frag_list]
    # Add the input molecule to the fragment list
    frag_smiles_list.append([Chem.MolToSmiles(mol), mol.GetNumAtoms(), 1])
    # Put the results into a Pandas dataframe
    frag_df = pd.DataFrame(frag_smiles_list, columns=["Scaffold", "NumAtoms", "NumRgroups"])
    # Remove duplicate fragments
    frag_df = frag_df.drop_duplicates("Scaffold")
    return frag_df

def find_scaffolds(df_in, progress_callback=None):
    """
    Generate scaffolds for a set of molecules
    :param df_in: Pandas dataframe with [SMILES, Name, RDKit molecule] columns
    :param progress_callback: function to update progress (optional)
    :return: dataframe with molecules and scaffolds, dataframe with unique scaffolds
    """
    # Loop over molecules and generate fragments
    df_list = []
    total_molecules = len(df_in)
    
    for idx, (smiles, name, mol) in enumerate(df_in[["SMILES", "Name", "mol"]].values):
        try:
            tmp_df = generate_fragments(mol).copy()
            tmp_df['Name'] = name
            tmp_df['SMILES'] = smiles
            tmp_df['Mol'] = [mol] * len(tmp_df)  # Store the molecule for later use
            df_list.append(tmp_df)
        except Exception as e:
            st.warning(f"Error processing molecule {name}: {e}")
        
        # Update progress if callback provided
        if progress_callback:
            progress_callback((idx + 1) / total_molecules)
    
    # Combine the list of dataframes into a single dataframe
    if df_list:
        mol_df = pd.concat(df_list, ignore_index=True)
    else:
        return pd.DataFrame(), pd.DataFrame()
    
    # Collect scaffolds
    scaffold_list = []
    for scaffold, group in mol_df.groupby("Scaffold"):
        unique_names = group['Name'].nunique()
        num_atoms = group['NumAtoms'].iloc[0]
        num_rgroups = group['NumRgroups'].iloc[0]
        
        # Get example molecules for this scaffold
        example_molecules = group['Mol'].iloc[:3].tolist()
        
        scaffold_list.append([scaffold, unique_names, num_atoms, num_rgroups, example_molecules])
    
    scaffold_df = pd.DataFrame(scaffold_list, 
                              columns=["Scaffold", "Count", "NumAtoms", "NumRgroups", "ExampleMolecules"])
    
    # Any fragment that occurs more times than the number of fragments can't be a scaffold
    # (Actually, we want to keep all fragments for MMP analysis)
    # Sort scaffolds by frequency and size
    scaffold_df = scaffold_df.sort_values(["Count", "NumAtoms"], ascending=[False, False])
    
    return mol_df, scaffold_df

def get_molecules_with_scaffold(scaffold, mol_df, activity_df):
    """
    Associate molecules with scaffolds using R-group decomposition
    :param scaffold: scaffold SMILES
    :param mol_df: dataframe with molecules and scaffolds
    :param activity_df: dataframe with [SMILES, Name, pIC50] columns
    :return: list of core(s) with R-groups labeled, dataframe with [SMILES, Name, pIC50, mol]
    """
    # Find molecules that contain this scaffold
    match_df = mol_df.query("Scaffold == @scaffold")
    
    if len(match_df) == 0:
        return [], pd.DataFrame()
    
    # Merge with activity data
    merge_df = match_df.merge(activity_df, on=["SMILES", "Name"])
    
    if len(merge_df) == 0:
        return [], pd.DataFrame()
    
    # Create scaffold molecule
    scaffold_mol = Chem.MolFromSmiles(scaffold)
    
    if scaffold_mol is None:
        return [], merge_df[["SMILES", "Name", "pIC50", "Mol"]]
    
    # Perform R-group decomposition
    try:
        # Prepare molecules for R-group decomposition
        molecules = []
        valid_indices = []
        
        for idx, mol in enumerate(merge_df['Mol']):
            if mol is not None:
                molecules.append(mol)
                valid_indices.append(idx)
        
        if molecules:
            rgroup_match, rgroup_miss = RGroupDecompose([scaffold_mol], molecules, asSmiles=True)
            
            if len(rgroup_match):
                rgroup_df = pd.DataFrame(rgroup_match)
                
                # Extract R-group information
                rgroup_columns = [col for col in rgroup_df.columns if col.startswith('R')]
                
                # Add SMILES and names back
                for i, idx in enumerate(valid_indices):
                    if i < len(rgroup_df):
                        rgroup_df.loc[i, 'SMILES'] = merge_df.iloc[idx]['SMILES']
                        rgroup_df.loc[i, 'Name'] = merge_df.iloc[idx]['Name']
                        rgroup_df.loc[i, 'pIC50'] = merge_df.iloc[idx]['pIC50']
                
                # Get unique cores
                unique_cores = rgroup_df['Core'].unique()
                
                return unique_cores, rgroup_df
    except Exception as e:
        st.warning(f"R-group decomposition error: {e}")
    
    # Fallback: return basic info
    return [], merge_df[["SMILES", "Name", "pIC50", "Mol"]]

def highlight_common_scaffold(mol1, mol2, highlight_color=(0.8, 0.8, 0.8, 0.6)):
    """Highlight the common scaffold between two molecules"""
    try:
        # Find maximum common substructure
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

def draw_molecule_pair(mol1, mol2, name1="Compound 1", name2="Compound 2", 
                      pIC50_1=None, pIC50_2=None, highlight_core=True, size=(300, 300)):
    """Draw two molecules side by side with highlighting"""
    
    # Prepare molecules for display
    mols = [mol1, mol2]
    
    # Highlight common core if requested
    highlight_atoms = []
    
    if highlight_core and mol1 and mol2:
        atoms1, atoms2 = highlight_common_scaffold(mol1, mol2)
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
                             highlightAtomLists=highlight_atoms if highlight_atoms else None)
        return img
    except:
        # Fallback to simple drawing
        img = MolsToGridImage(mols, molsPerRow=2, subImgSize=size, legends=labels)
        return img

def parse_transform_to_smiles(transform_str):
    """Convert a transformation string to proper SMILES notation."""
    
    if '>>' not in transform_str:
        return None, None, None
    
    reactant_str, product_str = transform_str.split('>>')
    
    # Parse dot-separated fragments
    def fragments_to_smiles(fragment_str):
        fragments = fragment_str.split('.')
        return fragments
    
    reactant_fragments = fragments_to_smiles(reactant_str)
    product_fragments = fragments_to_smiles(product_str)
    
    # Find the largest common fragment (likely the core)
    common_fragments = list(set(reactant_fragments) & set(product_fragments))
    
    # Get changing fragments (R-groups)
    reactant_r_groups = [f for f in reactant_fragments if f not in common_fragments]
    product_r_groups = [f for f in product_fragments if f not in common_fragments]
    
    # Create full molecules by combining
    if common_fragments:
        # Use the largest common fragment as core
        common_core = max(common_fragments, key=len) if common_fragments else ""
        
        # Create reactant molecule: core + R-groups
        reactant_full = common_core
        if reactant_r_groups:
            reactant_full = f"{common_core}.{'.'.join(reactant_r_groups)}"
        
        # Create product molecule: core + R-groups
        product_full = common_core
        if product_r_groups:
            product_full = f"{common_core}.{'.'.join(product_r_groups)}"
        
        # Create simplified transformation
        reactant_simple = '.'.join(reactant_r_groups) if reactant_r_groups else "*"
        product_simple = '.'.join(product_r_groups) if product_r_groups else "*"
        
        return reactant_full, product_full, f"{reactant_simple}>>{product_simple}"
    
    # If no common fragments found, use the full strings
    return reactant_str, product_str, transform_str

def visualize_transformation_enhanced(transform_str, size=(400, 200)):
    """Create an enhanced visualization of the transformation with proper chemical structures"""
    
    # Try to parse the transformation to proper SMILES
    reactant_smiles, product_smiles, simple_transform = parse_transform_to_smiles(transform_str)
    
    try:
        # Create reactant and product molecules
        reactant_mol = Chem.MolFromSmiles(reactant_smiles) if reactant_smiles else None
        product_mol = Chem.MolFromSmiles(product_smiles) if product_smiles else None
        
        if reactant_mol and product_mol:
            # Both molecules are valid - draw them side by side
            mols = [reactant_mol, product_mol]
            legends = ["Reactant (Before)", "Product (After)"]
            
            # Draw the molecules
            img = MolsToGridImage(mols, molsPerRow=2, subImgSize=(size[0]//2, size[1]), 
                                 legends=legends)
            return img
        
        elif reactant_mol or product_mol:
            # Only one is valid
            valid_mols = []
            valid_legends = []
            
            if reactant_mol:
                valid_mols.append(reactant_mol)
                valid_legends.append("Reactant (Before)")
            
            if product_mol:
                valid_mols.append(product_mol)
                valid_legends.append("Product (After)")
            
            img = MolsToGridImage(valid_mols, molsPerRow=len(valid_mols), 
                                 subImgSize=(size[0]//len(valid_mols), size[1]), 
                                 legends=valid_legends)
            return img
        
        else:
            # Try to create from simple fragments
            if '>>' in transform_str:
                reactant_part, product_part = transform_str.split('>>')
                
                # Try to parse individual fragments
                reactant_frags = reactant_part.split('.')
                product_frags = product_part.split('.')
                
                # Try to create molecules from each fragment
                reactant_mols = []
                product_mols = []
                
                for frag in reactant_frags[:3]:  # Limit to first 3 fragments
                    mol = Chem.MolFromSmiles(frag)
                    if mol:
                        reactant_mols.append(mol)
                
                for frag in product_frags[:3]:  # Limit to first 3 fragments
                    mol = Chem.MolFromSmiles(frag)
                    if mol:
                        product_mols.append(mol)
                
                if reactant_mols or product_mols:
                    # Combine all fragments
                    all_mols = reactant_mols + product_mols
                    all_legends = ["Fragment"] * len(all_mols)
                    
                    if len(all_mols) > 0:
                        n_cols = min(4, len(all_mols))
                        img = MolsToGridImage(all_mols, molsPerRow=n_cols, 
                                             subImgSize=(200, 200), 
                                             legends=all_legends)
                        return img
            
            # Fallback to text visualization
            return create_text_visualization(transform_str, size)
            
    except Exception as e:
        # Create text-based visualization as fallback
        return create_text_visualization(f"{transform_str[:50]}...", size)

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
            
            # Parse and display the transformation in a better format
            reactant_smiles, product_smiles, simple_transform = parse_transform_to_smiles(transform_str)
            
            st.markdown("**Full Transformation:**")
            if show_smiles:
                # Display in a more readable format
                st.code(f"Before: {reactant_smiles[:100]}...", language="text")
                st.code(f"After:  {product_smiles[:100]}...", language="text")
            
            # Show molecule visualization if enabled
            if show_molecules:
                st.markdown("**Structural Change:**")
                img = visualize_transformation_enhanced(transform_str, size=(image_size, image_size//2))
                st.image(img, caption="Transformation Visualization")
        
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
                
                # Also show the specific transformation for this pair
                st.markdown("**Transformation for this pair:**")
                if 'Transform' in pair:
                    img = visualize_transformation_enhanced(pair['Transform'], size=(400, 150))
                    st.image(img, caption="Specific transformation")
    
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

def run_scaffold_analysis(df, min_scaffold_count=2, min_fraction=0.67):
    """Run scaffold analysis using the improved scaffold finder"""
    
    # Process molecules
    df['mol'] = df['SMILES'].apply(Chem.MolFromSmiles)
    valid_df = df[df['mol'].notna()].copy()
    
    if len(valid_df) == 0:
        st.error("No valid molecules found in the dataset.")
        return None, None
    
    st.info(f"✅ Found {len(valid_df)} valid molecules out of {len(df)} total.")
    
    # Display progress
    progress_bar = st.progress(0)
    
    def update_progress(progress):
        progress_bar.progress(progress)
    
    # Find scaffolds
    st.info("🔬 Finding scaffolds...")
    mol_df, scaffold_df = find_scaffolds(valid_df, progress_callback=update_progress)
    
    if len(mol_df) == 0 or len(scaffold_df) == 0:
        st.error("No scaffolds found.")
        return None, None
    
    # Filter scaffolds by minimum count
    filtered_scaffolds = scaffold_df[scaffold_df['Count'] >= min_scaffold_count]
    
    if len(filtered_scaffolds) == 0:
        st.warning(f"No scaffolds found with at least {min_scaffold_count} occurrences.")
        st.info("Try lowering the minimum scaffold count.")
        return scaffold_df, mol_df
    
    st.success(f"✅ Found {len(filtered_scaffolds)} scaffolds with at least {min_scaffold_count} occurrences.")
    
    return filtered_scaffolds, mol_df

def run_mmp_analysis_from_scaffolds(scaffold_df, mol_df, activity_df, min_occurrences=2):
    """Run MMP analysis based on identified scaffolds"""
    
    if len(scaffold_df) == 0 or len(mol_df) == 0:
        st.error("No scaffolds or molecules to analyze.")
        return None, None, None
    
    delta_list = []
    
    # For each scaffold, find molecular pairs
    for scaffold in scaffold_df['Scaffold'].head(10):  # Limit to top 10 scaffolds for performance
        try:
            # Get molecules with this scaffold
            cores, rgroup_df = get_molecules_with_scaffold(scaffold, mol_df, activity_df)
            
            if len(rgroup_df) < 2:
                continue
            
            # Get R-group columns
            rgroup_columns = [col for col in rgroup_df.columns if col.startswith('R')]
            
            if len(rgroup_columns) == 0:
                continue
            
            # Create pairs based on R-group differences
            for i in range(len(rgroup_df)):
                for j in range(i+1, len(rgroup_df)):
                    row_i = rgroup_df.iloc[i]
                    row_j = rgroup_df.iloc[j]
                    
                    # Check if they differ in exactly one R-group
                    differing_rgroups = []
                    for rg_col in rgroup_columns:
                        if row_i[rg_col] != row_j[rg_col]:
                            differing_rgroups.append(rg_col)
                    
                    if len(differing_rgroups) == 1:
                        # This is a valid MMP pair
                        rg_col = differing_rgroups[0]
                        transform = f"{row_i[rg_col]}>>{row_j[rg_col]}"
                        delta = row_j['pIC50'] - row_i['pIC50']
                        
                        delta_list.append([
                            row_i['SMILES'], scaffold, row_i[rg_col], row_i['Name'], row_i['pIC50'],
                            row_j['SMILES'], scaffold, row_j[rg_col], row_j['Name'], row_j['pIC50'],
                            transform, delta
                        ])
        
        except Exception as e:
            st.warning(f"Error processing scaffold {scaffold[:50]}: {e}")
            continue
    
    if not delta_list:
        st.warning("No MMP pairs found from scaffolds.")
        return None, None, None
    
    cols = ["SMILES_1", "Core_1", "R_group_1", "Name_1", "pIC50_1",
           "SMILES_2", "Core_2", "R_group_2", "Name_2", "pIC50_2",
           "Transform", "Delta"]
    delta_df = pd.DataFrame(delta_list, columns=cols)
    
    # Aggregate transformations
    mmp_list = []
    for transform, group in delta_df.groupby("Transform"):
        if len(group) >= min_occurrences:
            mmp_list.append([transform, len(group), group.Delta.values])
    
    if not mmp_list:
        st.warning(f"No transformations found with at least {min_occurrences} occurrences.")
        return delta_df, pd.DataFrame()
    
    mmp_df = pd.DataFrame(mmp_list, columns=["Transform", "Count", "Deltas"])
    mmp_df['mean_delta'] = [x.mean() for x in mmp_df.Deltas]
    mmp_df['std_delta'] = [x.std() for x in mmp_df.Deltas]
    
    return delta_df, mmp_df

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

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
st.sidebar.header("⚙️ Analysis Parameters")
analysis_mode = st.sidebar.radio(
    "Analysis Mode",
    ["MMP Analysis Only", "Scaffold Finder Only", "Both MMP & Scaffold Analysis"],
    index=2,
    help="Choose the analysis mode"
)

min_scaffold_count = st.sidebar.slider(
    "Minimum scaffold occurrences",
    min_value=1,
    max_value=20,
    value=2,
    help="Minimum number of compounds sharing a scaffold"
)

min_fragment_fraction = st.sidebar.slider(
    "Minimum fragment fraction",
    min_value=0.1,
    max_value=1.0,
    value=0.67,
    step=0.05,
    help="Minimum fraction of atoms in scaffold (relative to original molecule)"
)

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

# ============================================================================
# MAIN APP LOGIC
# ============================================================================

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
        
        col_preview1, col_preview2 = st.columns([1, 2])
        
        with col_preview1:
            st.dataframe(df.head(), use_container_width=True)
            st.write(f"Total compounds: {len(df)}")
        
        with col_preview2:
            # Basic statistics
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
        
        # Process molecules
        df['mol'] = df['SMILES'].apply(Chem.MolFromSmiles)
        valid_df = df[df['mol'].notna()].copy()
        
        if len(valid_df) == 0:
            st.error("No valid molecules found in the dataset.")
            st.stop()
        
        # Run selected analysis
        if analysis_mode in ["Scaffold Finder Only", "Both MMP & Scaffold Analysis"]:
            st.header("🏗️ Scaffold Finder Analysis")
            
            with st.spinner("Finding scaffolds..."):
                scaffold_df, mol_df = run_scaffold_analysis(
                    valid_df, 
                    min_scaffold_count=min_scaffold_count,
                    min_fraction=min_fragment_fraction
                )
            
            if scaffold_df is not None and len(scaffold_df) > 0:
                # Display top scaffolds
                st.subheader(f"Top {min(10, len(scaffold_df))} Scaffolds")
                
                # Create a nice display of scaffolds
                display_scaffold_df = scaffold_df.head(10).copy()
                display_scaffold_df['Scaffold Image'] = ""  # Placeholder for images
                
                # Display scaffold table with images
                for idx, row in display_scaffold_df.iterrows():
                    with st.expander(f"Scaffold {idx+1}: {row['Scaffold'][:50]}... (Count: {row['Count']})", expanded=(idx==0)):
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            # Display scaffold structure
                            scaffold_mol = Chem.MolFromSmiles(row['Scaffold'])
                            if scaffold_mol:
                                img = Draw.MolToImage(scaffold_mol, size=(300, 300))
                                st.image(img, caption=f"Scaffold {idx+1}")
                            
                            st.write(f"**SMILES:** {row['Scaffold']}")
                            st.write(f"**Number of Atoms:** {row['NumAtoms']}")
                            st.write(f"**Number of R-groups:** {row['NumRgroups']}")
                        
                        with col2:
                            # Display compounds with this scaffold
                            match_df = mol_df[mol_df['Scaffold'] == row['Scaffold']]
                            st.write(f"**Compounds with this scaffold ({len(match_df)}):**")
                            
                            # Show compounds in a grid
                            compound_cols = st.columns(3)
                            for i, (_, compound_row) in enumerate(match_df.head(6).iterrows()):
                                with compound_cols[i % 3]:
                                    mol = Chem.MolFromSmiles(compound_row['SMILES'])
                                    if mol:
                                        img = Draw.MolToImage(mol, size=(150, 150))
                                        st.image(img, caption=f"{compound_row['Name']}\npIC50: {compound_row['pIC50']:.2f}")
                            
                            if len(match_df) > 6:
                                st.write(f"... and {len(match_df) - 6} more compounds")
                
                # MMP analysis based on scaffolds
                if analysis_mode == "Both MMP & Scaffold Analysis" and len(scaffold_df) > 0:
                    st.header("🔄 MMP Analysis from Scaffolds")
                    
                    with st.spinner("Running MMP analysis from scaffolds..."):
                        delta_df, mmp_df = run_mmp_analysis_from_scaffolds(
                            scaffold_df, 
                            mol_df, 
                            valid_df[['SMILES', 'Name', 'pIC50', 'mol']],
                            min_occurrences=min_transform_occurrence
                        )
                    
                    if delta_df is not None and mmp_df is not None and len(mmp_df) > 0:
                        display_mmp_results(mmp_df, delta_df, valid_df)
                    else:
                        st.warning("No significant MMP transformations found from scaffolds.")
        
        # Original MMP analysis (if selected)
        if analysis_mode in ["MMP Analysis Only", "Both MMP & Scaffold Analysis"]:
            if analysis_mode == "MMP Analysis Only":
                st.header("🔄 Original MMP Analysis")
            else:
                st.header("🔄 Additional MMP Analysis (Original Method)")
            
            # Run original MMP analysis (simplified version)
            with st.spinner("Running original MMP analysis..."):
                # Simplified MMP analysis
                delta_list = []
                
                # Generate fragments for each molecule
                progress_bar = st.progress(0)
                for idx, (smiles, name, pIC50, mol) in enumerate(valid_df[['SMILES', 'Name', 'pIC50', 'mol']].values):
                    try:
                        # Generate fragments
                        frag_df = generate_fragments(mol)
                        
                        for _, frag_row in frag_df.iterrows():
                            delta_list.append([
                                smiles, frag_row['Scaffold'], name, pIC50,
                                frag_row['NumAtoms'], frag_row['NumRgroups']
                            ])
                    except Exception as e:
                        st.warning(f"Error processing molecule {name}: {e}")
                    
                    progress_bar.progress((idx + 1) / len(valid_df))
                
                if delta_list:
                    frag_df = pd.DataFrame(delta_list, columns=[
                        "SMILES", "Scaffold", "Name", "pIC50", "NumAtoms", "NumRgroups"
                    ])
                    
                    # Find pairs with same scaffold
                    mmp_pairs = []
                    for scaffold, group in frag_df.groupby("Scaffold"):
                        if len(group) > 1:
                            compounds = group.to_dict('records')
                            
                            for i in range(len(compounds)):
                                for j in range(i+1, len(compounds)):
                                    comp_a = compounds[i]
                                    comp_b = compounds[j]
                                    
                                    if comp_a['SMILES'] != comp_b['SMILES']:
                                        delta = comp_b['pIC50'] - comp_a['pIC50']
                                        transform = f"{comp_a['Name']}>>{comp_b['Name']}"
                                        
                                        mmp_pairs.append([
                                            comp_a['SMILES'], comp_a['Scaffold'], comp_a['Name'], comp_a['pIC50'],
                                            comp_b['SMILES'], comp_b['Scaffold'], comp_b['Name'], comp_b['pIC50'],
                                            transform, delta
                                        ])
                    
                    if mmp_pairs:
                        mmp_cols = ["SMILES_1", "Scaffold_1", "Name_1", "pIC50_1",
                                  "SMILES_2", "Scaffold_2", "Name_2", "pIC50_2",
                                  "Transform", "Delta"]
                        mmp_result_df = pd.DataFrame(mmp_pairs, columns=mmp_cols)
                        
                        # Group by transform pattern
                        transform_stats = []
                        for transform, group in mmp_result_df.groupby("Transform"):
                            if len(group) >= min_transform_occurrence:
                                transform_stats.append([
                                    transform, len(group), 
                                    group['Delta'].mean(), group['Delta'].std()
                                ])
                        
                        if transform_stats:
                            transform_df = pd.DataFrame(transform_stats, 
                                                      columns=["Transform", "Count", "MeanDelta", "StdDelta"])
                            transform_df = transform_df.sort_values("MeanDelta", ascending=False)
                            
                            st.success(f"✅ Found {len(transform_df)} significant transformations.")
                            
                            # Display top transforms
                            st.subheader(f"Top {min(num_top_transforms, len(transform_df))} Transformations")
                            
                            for i, (_, row) in enumerate(transform_df.head(num_top_transforms).iterrows()):
                                with st.expander(f"Transformation {i+1}: ΔpIC50 = {row['MeanDelta']:.3f} ± {row['StdDelta']:.3f}", 
                                               expanded=(i==0)):
                                    col1, col2 = st.columns([1, 2])
                                    
                                    with col1:
                                        st.write(f"**Transformation:**")
                                        st.code(row['Transform'])
                                        st.write(f"**Occurrences:** {row['Count']}")
                                        st.write(f"**Mean ΔpIC50:** {row['MeanDelta']:.3f}")
                                        st.write(f"**Std ΔpIC50:** {row['StdDelta']:.3f}")
                                    
                                    with col2:
                                        # Show example pairs
                                        example_pairs = mmp_result_df[
                                            mmp_result_df['Transform'] == row['Transform']
                                        ].head(3)
                                        
                                        for _, pair in example_pairs.iterrows():
                                            col_a, col_b = st.columns(2)
                                            
                                            with col_a:
                                                mol_a = Chem.MolFromSmiles(pair['SMILES_1'])
                                                if mol_a:
                                                    img = Draw.MolToImage(mol_a, size=(150, 150))
                                                    st.image(img, 
                                                           caption=f"{pair['Name_1']}\npIC50: {pair['pIC50_1']:.2f}")
                                            
                                            with col_b:
                                                mol_b = Chem.MolFromSmiles(pair['SMILES_2'])
                                                if mol_b:
                                                    img = Draw.MolToImage(mol_b, size=(150, 150))
                                                    st.image(img, 
                                                           caption=f"{pair['Name_2']}\npIC50: {pair['pIC50_2']:.2f}")
                                            
                                            delta_val = pair['pIC50_2'] - pair['pIC50_1']
                                            color = "green" if delta_val > 0 else "red" if delta_val < 0 else "gray"
                                            st.markdown(f"<p style='color:{color}; text-align:center'>ΔpIC50 = {delta_val:.3f}</p>", 
                                                       unsafe_allow_html=True)
                                            st.markdown("---")
                        else:
                            st.warning("No significant transformations found.")
                    else:
                        st.warning("No molecular pairs found.")
                else:
                    st.warning("No fragments generated.")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

else:
    # Show instructions when no file is uploaded
    st.info("👈 Please upload a CSV file to begin analysis")
    
    # Show example data
    st.header("📋 Example Data Format")
    st.markdown("""
    **For best results with scaffold finder, use data like this:**
    
    Compounds that share a common scaffold with different R-groups.
    """)
    
    example_data = pd.DataFrame({
        'SMILES': [
            'Cc1ccccc1C(=O)O',      # Benzoic acid with methyl
            'COc1ccccc1C(=O)O',     # Benzoic acid with methoxy
            'FC1=C(C(=O)O)C=CC=C1', # Benzoic acid with fluoro
            'Clc1ccccc1C(=O)O',     # Benzoic acid with chloro
            'BrC1=CC=CC=C1C(=O)O',  # Benzoic acid with bromo
            'IC1=CC=CC=C1C(=O)O',   # Benzoic acid with iodo
            'OC1=CC=CC=C1C(=O)O',   # Benzoic acid with hydroxy
            'NC1=CC=CC=C1C(=O)O',   # Benzoic acid with amino
        ],
        'Name': ['Methyl_BA', 'Methoxy_BA', 'Fluoro_BA', 'Chloro_BA', 
                'Bromo_BA', 'Iodo_BA', 'Hydroxy_BA', 'Amino_BA'],
        'pIC50': [7.2, 7.8, 6.9, 7.5, 7.1, 6.8, 8.0, 6.5]
    })
    
    st.dataframe(example_data)
    
    # Display example molecules
    if show_molecules:
        st.subheader("Example Molecules")
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

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit & RDKit | MMP Analysis & Scaffold Finder</p>
    </div>
    """,
    unsafe_allow_html=True
)
