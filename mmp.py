# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import itertools
import sys
import warnings
from operator import itemgetter
import traceback

# Suppress warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="MMP Analysis Tool - Kekulization Fixed",
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
.warning-box {
    background-color: #FEF3C7;
    border-left: 4px solid #F59E0B;
    padding: 1rem;
    border-radius: 5px;
    margin: 1rem 0;
}
.success-box {
    background-color: #D1FAE5;
    border-left: 4px solid #10B981;
    padding: 1rem;
    border-radius: 5px;
    margin: 1rem 0;
}
.info-box {
    background-color: #E0F2FE;
    border-left: 4px solid #0EA5E9;
    padding: 1rem;
    border-radius: 5px;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🧪 MMP Analysis Tool - Kekulization Fixed</h1>', unsafe_allow_html=True)

# Try to import RDKit with error handling
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.Chem.rdmolops import ReplaceCore, RemoveHs
    from rdkit.Chem.rdchem import Mol
    from rdkit import RDLogger
    
    # Suppress RDKit warnings
    RDLogger.DisableLog('rdApp.*')
    
    RDKIT_AVAILABLE = True
except ImportError as e:
    st.error(f"RDKit not available: {e}")
    st.info("Please install RDKit with: pip install rdkit-pypi")
    RDKIT_AVAILABLE = False
except Exception as e:
    st.error(f"Error loading RDKit: {e}")
    st.warning("""
    **Potential Compatibility Issue**
    1. Try: `pip install "numpy<2" "rdkit-pypi"`
    2. Or use conda: `conda install -c conda-forge rdkit`
    """)
    RDKIT_AVAILABLE = False

# Sidebar
with st.sidebar:
    st.markdown("## 📋 Configuration")
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file and RDKIT_AVAILABLE:
        st.markdown("### ⚙️ Analysis Parameters")
        
        # Core parameters
        min_occurrence = st.slider(
            "Minimum transform occurrences", 
            1, 50, 3,
            help="Minimum number of occurrences for a transform to be considered"
        )
        
        # Fragmentation parameters
        st.markdown("### ✂️ Fragmentation Settings")
        
        max_cuts = st.slider(
            "Maximum cuts per molecule",
            1, 5, 2,
            help="Maximum number of bonds to cut (higher = more fragments but slower)"
        )
        
        min_fragment_size = st.slider(
            "Minimum fragment size (atoms)",
            1, 20, 5,
            help="Minimum atoms in fragment"
        )
        
        max_fragment_size = st.slider(
            "Maximum fragment size (atoms)",
            10, 100, 50,
            help="Maximum atoms in fragment"
        )
        
        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            use_smart_fragmentation = st.checkbox(
                "Use smart fragmentation", 
                value=True,
                help="Use RDKit's FragmentMol for better fragmentation"
            )
            
            fragment_single_bonds_only = st.checkbox(
                "Cut single bonds only", 
                value=True,
                help="Only cut single bonds (more stable)"
            )
            
            exclude_rings_from_cuts = st.checkbox(
                "Exclude ring bonds", 
                value=True,
                help="Don't cut bonds that are part of rings"
            )
            
            sanitize_molecules = st.checkbox(
                "Sanitize molecules", 
                value=True,
                help="Clean and validate molecules (recommended)"
            )
            
            # NEW: Handle kekulization issues
            handle_kekulization = st.checkbox(
                "Handle kekulization errors", 
                value=True,
                help="Skip kekulization for problematic molecules"
            )
            
            try_kekulize_on_failure = st.checkbox(
                "Try kekulization with fallback", 
                value=False,
                help="Attempt kekulization but recover if it fails"
            )
        
        # Display options
        st.markdown("### 👀 Display Options")
        transforms_to_display = st.slider(
            "Transforms to display", 
            1, 100, 20
        )
        
        show_debug = st.checkbox("Show debug information", value=False)

# Helper functions (only if RDKit is available)
if RDKIT_AVAILABLE:
    def safe_sanitize_mol(mol, sanitize=True, kekulize=False, handle_kekulization=True):
        """Safely sanitize a molecule with error handling for kekulization"""
        if mol is None:
            return None
        
        try:
            if sanitize:
                Chem.SanitizeMol(mol)
            
            if kekulize:
                if handle_kekulization:
                    try:
                        # Try to kekulize
                        Chem.Kekulize(mol, clearAromaticFlags=True)
                    except:
                        # If kekulization fails, just keep aromatic flags
                        st.warning("Kekulization failed - keeping aromatic flags")
                        pass
                else:
                    Chem.Kekulize(mol, clearAromaticFlags=True)
            
            return mol
        except Exception as e:
            if show_debug:
                st.warning(f"Sanitization error: {str(e)}")
            return mol
    
    def get_largest_fragment(mol):
        """Get the largest fragment from a molecule"""
        if mol is None:
            return None
        try:
            frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
            if not frags:
                return mol
            return max(frags, key=lambda x: x.GetNumAtoms())
        except:
            return mol
    
    def remove_atom_map_nums(mol):
        """Remove atom map numbers from a molecule"""
        if mol is None:
            return None
        try:
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
            return mol
        except:
            return mol
    
    def generate_fragments_rdkit(mol, max_cuts=2, min_fragment_size=5, max_fragment_size=50):
        """Generate fragments using RDKit's FragmentMol for smarter fragmentation"""
        if mol is None:
            return []
        
        try:
            # Use RDKit's FragmentMol for smarter fragmentation
            from rdkit.Chem.rdMMPA import FragmentMol
            
            # Generate fragments
            frags = FragmentMol(mol, maxCuts=max_cuts)
            
            # Flatten and process fragments
            all_frags = []
            for frag_list in frags:
                for frag in frag_list:
                    if frag is not None:
                        # Clean up fragment
                        frag = remove_atom_map_nums(frag)
                        
                        # Get largest fragment (in case of disconnected structures)
                        frag = get_largest_fragment(frag)
                        
                        # Check size constraints
                        num_atoms = frag.GetNumAtoms()
                        if min_fragment_size <= num_atoms <= max_fragment_size:
                            # Sanitize without kekulization to avoid errors
                            try:
                                Chem.SanitizeMol(frag)
                            except:
                                pass
                            all_frags.append(frag)
            
            # Remove duplicates by SMILES
            unique_frags = []
            seen_smiles = set()
            for frag in all_frags:
                try:
                    smiles = Chem.MolToSmiles(frag)
                    if smiles not in seen_smiles:
                        seen_smiles.add(smiles)
                        unique_frags.append(frag)
                except:
                    continue
            
            return unique_frags
            
        except Exception as e:
            if show_debug:
                st.warning(f"FragmentMol failed: {str(e)} - using fallback")
            return generate_fragments_simple(mol, max_cuts, min_fragment_size, max_fragment_size)
    
    def generate_fragments_simple(mol, max_cuts=2, min_fragment_size=5, max_fragment_size=50):
        """Simple bond cutting approach as fallback"""
        if mol is None:
            return []
        
        fragments = []
        num_atoms = mol.GetNumAtoms()
        
        # Always include the whole molecule
        if min_fragment_size <= num_atoms <= max_fragment_size:
            fragments.append(mol)
        
        if max_cuts > 0:
            # Find cuttable bonds
            cuttable_bonds = []
            for bond in mol.GetBonds():
                if fragment_single_bonds_only and bond.GetBondType() != Chem.BondType.SINGLE:
                    continue
                if exclude_rings_from_cuts and bond.IsInRing():
                    continue
                
                # Don't cut bonds to terminal atoms
                begin_atom = bond.GetBeginAtom()
                end_atom = bond.GetEndAtom()
                if begin_atom.GetDegree() == 1 or end_atom.GetDegree() == 1:
                    continue
                
                cuttable_bonds.append(bond.GetIdx())
            
            # Try cutting bonds
            for bond_idx in cuttable_bonds[:10]:  # Limit to first 10 bonds for performance
                try:
                    # Create editable molecule and remove bond
                    emol = Chem.EditableMol(mol)
                    bond = mol.GetBondWithIdx(bond_idx)
                    emol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                    frag_mol = emol.GetMol()
                    
                    # Get fragments
                    frags = Chem.GetMolFrags(frag_mol, asMols=True, sanitizeFrags=False)
                    
                    for frag in frags:
                        frag = remove_atom_map_nums(frag)
                        num_frag_atoms = frag.GetNumAtoms()
                        if min_fragment_size <= num_frag_atoms <= max_fragment_size:
                            try:
                                Chem.SanitizeMol(frag)
                            except:
                                pass
                            fragments.append(frag)
                    
                except Exception as e:
                    if show_debug:
                        st.warning(f"Bond cutting failed: {str(e)}")
                    continue
        
        # Remove duplicates
        unique_frags = []
        seen_smiles = set()
        for frag in fragments:
            try:
                smiles = Chem.MolToSmiles(frag)
                if smiles not in seen_smiles:
                    seen_smiles.add(smiles)
                    unique_frags.append(frag)
            except:
                continue
        
        return unique_frags
    
    def load_and_process_data(file, sanitize=True, handle_kekulization=True):
        """Load and process chemical data with proper error handling"""
        df = pd.read_csv(file)
        
        # Check required columns
        required_cols = ['SMILES', 'pIC50']
        if not all(col in df.columns for col in required_cols):
            st.error(f"CSV must contain columns: {required_cols}")
            return None
        
        # Add Name column if not present
        if 'Name' not in df.columns:
            df['Name'] = [f"CMPD_{i}" for i in range(len(df))]
        
        # Process molecules
        molecules = []
        errors = []
        kekulization_errors = []
        
        for idx, row in df.iterrows():
            try:
                smiles = str(row['SMILES'])
                
                # Parse molecule
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    errors.append(f"Row {idx}: Invalid SMILES '{smiles}'")
                    molecules.append(None)
                    continue
                
                # Handle sanitization and kekulization carefully
                mol = safe_sanitize_mol(
                    mol, 
                    sanitize=sanitize, 
                    kekulize=False,  # Don't kekulize by default
                    handle_kekulization=handle_kekulization
                )
                
                # Get largest fragment (remove salts)
                mol = get_largest_fragment(mol)
                
                molecules.append(mol)
                
            except Exception as e:
                error_msg = f"Row {idx}: Error processing '{smiles[:50]}...' - {str(e)}"
                if "kekul" in str(e).lower():
                    kekulization_errors.append(error_msg)
                else:
                    errors.append(error_msg)
                molecules.append(None)
        
        df['mol'] = molecules
        
        # Show errors
        if errors or kekulization_errors:
            with st.expander("⚠️ Processing Summary", expanded=False):
                if kekulization_errors:
                    st.warning(f"Kekulization warnings: {len(kekulization_errors)}")
                    if show_debug:
                        for error in kekulization_errors[:5]:
                            st.text(error[:200])
                
                if errors:
                    st.error(f"Processing errors: {len(errors)}")
                    for error in errors[:5]:
                        st.text(error[:200])
                
                if len(errors) > 5 or len(kekulization_errors) > 5:
                    st.info(f"... and {max(0, len(errors)-5) + max(0, len(kekulization_errors)-5)} more warnings/errors")
        
        # Remove invalid molecules
        valid_df = df[df['mol'].notna()].copy()
        if len(valid_df) < len(df):
            st.warning(f"Removed {len(df) - len(valid_df)} rows with invalid/unprocessable molecules")
        
        if len(valid_df) == 0:
            st.error("No valid molecules could be processed. Please check your SMILES strings.")
            return None
        
        return valid_df
    
    def perform_mmp_analysis(df, min_occurrence=3, **kwargs):
        """Perform MMP analysis with proper error handling"""
        if df is None or len(df) == 0:
            return None, None
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Step 1/3: Fragmenting molecules...")
        progress_bar.progress(30)
        
        # Fragment molecules
        all_fragments = []
        for idx, row in df.iterrows():
            mol = row['mol']
            if mol is None:
                continue
            
            # Generate fragments based on user selection
            if use_smart_fragmentation:
                fragments = generate_fragments_rdkit(mol, **kwargs)
            else:
                fragments = generate_fragments_simple(mol, **kwargs)
            
            for frag in fragments:
                try:
                    frag_smiles = Chem.MolToSmiles(frag)
                    all_fragments.append({
                        'original_smiles': row['SMILES'],
                        'name': row['Name'],
                        'pIC50': row['pIC50'],
                        'fragment_smiles': frag_smiles,
                        'fragment_mol': frag,
                        'num_atoms': frag.GetNumAtoms()
                    })
                except:
                    continue
        
        if not all_fragments:
            st.error("No valid fragments generated")
            return None, None
        
        # Convert to DataFrame
        frag_df = pd.DataFrame(all_fragments)
        
        status_text.text("Step 2/3: Finding common fragments...")
        progress_bar.progress(60)
        
        # Group by fragment to find common cores
        common_cores = []
        for frag_smiles, group in frag_df.groupby('fragment_smiles'):
            if len(group) >= min_occurrence:
                common_cores.append({
                    'core_smiles': frag_smiles,
                    'count': len(group),
                    'compounds': list(group['original_smiles']),
                    'names': list(group['name']),
                    'pIC50_values': list(group['pIC50']),
                    'avg_pIC50': group['pIC50'].mean(),
                    'std_pIC50': group['pIC50'].std()
                })
        
        if not common_cores:
            st.warning(f"No common fragments found with {min_occurrence}+ occurrences")
            return frag_df, None
        
        # Sort by count
        common_cores_df = pd.DataFrame(common_cores)
        common_cores_df = common_cores_df.sort_values('count', ascending=False)
        
        status_text.text("Step 3/3: Analyzing transformations...")
        progress_bar.progress(90)
        
        # Find MMPs for each common core
        mmp_results = []
        
        for _, core_info in common_cores_df.iterrows():
            core_smiles = core_info['core_smiles']
            core_mol = Chem.MolFromSmiles(core_smiles)
            
            if core_mol is None:
                continue
            
            # Get compounds with this core
            core_compounds = frag_df[frag_df['fragment_smiles'] == core_smiles]
            
            if len(core_compounds) < 2:
                continue
            
            # Try to identify R-groups by removing core from molecules
            transformations = {}
            
            for _, row in core_compounds.iterrows():
                try:
                    original_mol = Chem.MolFromSmiles(row['original_smiles'])
                    if original_mol is None:
                        continue
                    
                    # Try to find what's different between original and core
                    # This is a simplified approach
                    original_smiles = Chem.MolToSmiles(original_mol)
                    core_smiles_simple = Chem.MolToSmiles(core_mol)
                    
                    if original_smiles != core_smiles_simple:
                        # Store transformation info
                        transform_key = f"{core_smiles_simple}>>{original_smiles}"
                        
                        if transform_key not in transformations:
                            transformations[transform_key] = {
                                'core': core_smiles_simple,
                                'r_group': original_smiles.replace(core_smiles_simple, '') if core_smiles_simple in original_smiles else 'UNKNOWN',
                                'compounds': [],
                                'pIC50_values': []
                            }
                        
                        transformations[transform_key]['compounds'].append(row['name'])
                        transformations[transform_key]['pIC50_values'].append(row['pIC50'])
                
                except Exception as e:
                    if show_debug:
                        st.warning(f"Transformation analysis failed: {str(e)}")
                    continue
            
            # Process transformations for this core
            for transform_key, transform_info in transformations.items():
                if len(transform_info['pIC50_values']) >= 2:
                    pIC50_values = transform_info['pIC50_values']
                    mean_pIC50 = np.mean(pIC50_values)
                    std_pIC50 = np.std(pIC50_values)
                    
                    mmp_results.append({
                        'core_smiles': transform_info['core'],
                        'transform': transform_key,
                        'r_group': transform_info['r_group'][:100],  # Limit length
                        'count': len(transform_info['pIC50_values']),
                        'mean_pIC50': mean_pIC50,
                        'std_pIC50': std_pIC50,
                        'min_pIC50': min(pIC50_values),
                        'max_pIC50': max(pIC50_values),
                        'range_pIC50': max(pIC50_values) - min(pIC50_values),
                        'compounds': ', '.join(transform_info['compounds'][:5])  # Limit
                    })
        
        status_text.text("Analysis complete!")
        progress_bar.progress(100)
        status_text.empty()
        progress_bar.empty()
        
        mmp_df = pd.DataFrame(mmp_results) if mmp_results else None
        
        return frag_df, mmp_df
    
    def create_transformation_plot(transform_data, title="Transformation Analysis"):
        """Create a plot for transformation data"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot 1: Distribution of counts
        if 'count' in transform_data.columns:
            axes[0].hist(transform_data['count'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0].set_xlabel('Number of Occurrences')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title('Distribution of Transform Occurrences')
            axes[0].grid(True, alpha=0.3)
        
        # Plot 2: pIC50 range vs count
        if 'range_pIC50' in transform_data.columns and 'count' in transform_data.columns:
            axes[1].scatter(transform_data['count'], transform_data['range_pIC50'], 
                          alpha=0.6, color='coral', s=50)
            axes[1].set_xlabel('Number of Occurrences')
            axes[1].set_ylabel('pIC50 Range')
            axes[1].set_title('Transform Impact vs Frequency')
            axes[1].grid(True, alpha=0.3)
            
            # Add trend line
            if len(transform_data) > 1:
                z = np.polyfit(transform_data['count'], transform_data['range_pIC50'], 1)
                p = np.poly1d(z)
                axes[1].plot(transform_data['count'], p(transform_data['count']), 
                           "r--", alpha=0.8, label='Trend')
                axes[1].legend()
        
        plt.tight_layout()
        return fig

# Main app logic
if not RDKIT_AVAILABLE:
    st.error("RDKit is not available. Please install it to use this tool.")
elif uploaded_file is not None:
    # Get parameters
    sanitize = sanitize_molecules if 'sanitize_molecules' in locals() else True
    handle_kekulization_val = handle_kekulization if 'handle_kekulization' in locals() else True
    
    # Load data with proper error handling
    with st.spinner("Loading and processing molecules..."):
        df = load_and_process_data(
            uploaded_file, 
            sanitize=sanitize,
            handle_kekulization=handle_kekulization_val
        )
    
    if df is not None and len(df) > 0:
        # Show dataset info
        st.markdown('<div class="success-box">✅ Data loaded successfully!</div>', unsafe_allow_html=True)
        
        with st.expander("📊 Dataset Overview", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Compounds", len(df))
            col2.metric("Min pIC50", f"{df['pIC50'].min():.2f}")
            col3.metric("Max pIC50", f"{df['pIC50'].max():.2f}")
            col4.metric("Avg pIC50", f"{df['pIC50'].mean():.2f}")
            
            # Show sample molecules
            st.subheader("Sample Molecules (First 4)")
            cols = st.columns(4)
            for idx, (_, row) in enumerate(df.head(4).iterrows()):
                with cols[idx]:
                    try:
                        mol = row['mol']
                        if mol:
                            img = Draw.MolToImage(mol, size=(200, 200))
                            name = row['Name']
                            pIC50 = row['pIC50']
                            st.image(img, caption=f"{name}\npIC50: {pIC50:.2f}")
                    except Exception as e:
                        st.error(f"Error displaying molecule: {str(e)}")
        
        # Perform MMP analysis
        st.markdown('<h2 class="section-header">🔍 MMP Analysis</h2>', unsafe_allow_html=True)
        
        # Get fragmentation parameters
        frag_params = {
            'max_cuts': max_cuts,
            'min_fragment_size': min_fragment_size,
            'max_fragment_size': max_fragment_size
        }
        
        # Check if use_smart_fragmentation is defined
        if 'use_smart_fragmentation' not in locals():
            use_smart_fragmentation = True
        
        with st.spinner("Performing MMP analysis..."):
            frag_df, mmp_df = perform_mmp_analysis(
                df, 
                min_occurrence=min_occurrence,
                **frag_params
            )
        
        if frag_df is not None:
            # Show fragment statistics
            st.markdown('<h3 class="section-header">📈 Fragment Statistics</h3>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Fragments", len(frag_df))
            col2.metric("Unique Fragments", frag_df['fragment_smiles'].nunique())
            col3.metric("Avg Fragment Size", f"{frag_df['num_atoms'].mean():.1f} atoms")
            col4.metric("Compounds with Fragments", frag_df['original_smiles'].nunique())
            
            # Show fragment size distribution
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(frag_df['num_atoms'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
            ax.set_xlabel('Number of Atoms in Fragment')
            ax.set_ylabel('Frequency')
            ax.set_title('Fragment Size Distribution')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        if mmp_df is not None and len(mmp_df) > 0:
            st.markdown('<div class="success-box">✅ Found {} significant transformations!</div>'.format(len(mmp_df)), unsafe_allow_html=True)
            
            # Show transformation statistics
            st.markdown('<h3 class="section-header">🔄 Transformation Statistics</h3>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Transforms", len(mmp_df))
            col2.metric("Avg Occurrences", f"{mmp_df['count'].mean():.1f}")
            
            # Find most impactful transforms
            if 'range_pIC50' in mmp_df.columns:
                max_range = mmp_df['range_pIC50'].max()
                col3.metric("Max pIC50 Range", f"{max_range:.2f}")
            
            # Count transforms with significant range
            if 'range_pIC50' in mmp_df.columns:
                sig_transforms = len(mmp_df[mmp_df['range_pIC50'] > 1.0])
                col4.metric("Transforms (ΔpIC50 > 1)", sig_transforms)
            
            # Create visualization
            st.markdown('<h4>Transformation Analysis</h4>', unsafe_allow_html=True)
            fig = create_transformation_plot(mmp_df)
            st.pyplot(fig)
            
            # Show top transformations
            st.markdown('<h3 class="section-header">🏆 Top Transformations</h3>', unsafe_allow_html=True)
            
            # Sort by impact (range) and then by count
            if 'range_pIC50' in mmp_df.columns:
                display_df = mmp_df.sort_values(['range_pIC50', 'count'], ascending=[False, False])
            else:
                display_df = mmp_df.sort_values('count', ascending=False)
            
            # Limit to requested number
            display_df = display_df.head(transforms_to_display)
            
            # Display in a nice table
            display_columns = ['transform', 'count', 'mean_pIC50', 'std_pIC50', 'range_pIC50']
            display_columns = [col for col in display_columns if col in display_df.columns]
            
            # Format the DataFrame for display
            formatted_df = display_df[display_columns].copy()
            if 'mean_pIC50' in formatted_df.columns:
                formatted_df['mean_pIC50'] = formatted_df['mean_pIC50'].round(2)
            if 'std_pIC50' in formatted_df.columns:
                formatted_df['std_pIC50'] = formatted_df['std_pIC50'].round(2)
            if 'range_pIC50' in formatted_df.columns:
                formatted_df['range_pIC50'] = formatted_df['range_pIC50'].round(2)
            
            # Rename columns for display
            column_names = {
                'transform': 'Transformation',
                'count': 'Count',
                'mean_pIC50': 'Mean pIC50',
                'std_pIC50': 'Std Dev',
                'range_pIC50': 'ΔpIC50 Range'
            }
            formatted_df = formatted_df.rename(columns=column_names)
            
            st.dataframe(formatted_df, use_container_width=True)
            
            # Show detailed view of top transformation
            if len(display_df) > 0:
                st.markdown('<h4>Detailed View of Top Transformation</h4>', unsafe_allow_html=True)
                
                top_transform = display_df.iloc[0]
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown("**Transformation Details:**")
                    st.text(f"Core: {top_transform['core_smiles'][:100]}...")
                    st.text(f"R-group: {top_transform['r_group'][:100]}...")
                    st.text(f"Occurrences: {top_transform['count']}")
                    if 'mean_pIC50' in top_transform:
                        st.text(f"Mean pIC50: {top_transform['mean_pIC50']:.2f}")
                    if 'range_pIC50' in top_transform:
                        st.text(f"pIC50 Range: {top_transform['range_pIC50']:.2f}")
                
                with col2:
                    # Try to display molecules
                    try:
                        # Parse core and try to display
                        core_mol = Chem.MolFromSmiles(str(top_transform['core_smiles']))
                        if core_mol:
                            img = Draw.MolToImage(core_mol, size=(300, 200))
                            st.image(img, caption="Core Structure")
                    except:
                        st.info("Could not display molecule structure")
            
            # Export options
            st.markdown('<h3 class="section-header">💾 Export Results</h3>', unsafe_allow_html=True)
            
            @st.cache_data
            def convert_df_to_csv(df):
                return df.to_csv(index=False).encode('utf-8')
            
            @st.cache_data
            def convert_dict_to_excel(data_dict):
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    for sheet_name, df in data_dict.items():
                        if df is not None and len(df) > 0:
                            df.to_excel(writer, index=False, sheet_name=sheet_name[:31])
                return output.getvalue()
            
            col1, col2 = st.columns(2)
            
            with col1:
                if mmp_df is not None:
                    st.download_button(
                        label="📥 Download Transformations (CSV)",
                        data=convert_df_to_csv(mmp_df),
                        file_name="mmp_transformations.csv",
                        mime="text/csv"
                    )
            
            with col2:
                export_data = {
                    'Transformations': mmp_df,
                    'Fragments': frag_df if frag_df is not None else pd.DataFrame(),
                    'Original_Data': df[['SMILES', 'Name', 'pIC50']]
                }
                
                excel_data = convert_dict_to_excel(export_data)
                st.download_button(
                    label="📥 Download Full Analysis (Excel)",
                    data=excel_data,
                    file_name="mmp_full_analysis.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            st.markdown('<div class="warning-box">⚠️ No significant transformations found. Try adjusting the parameters.</div>', unsafe_allow_html=True)
            
            if frag_df is not None:
                st.info(f"Generated {len(frag_df)} fragments but none occurred {min_occurrence}+ times.")
                st.info("Try:")
                st.markdown("1. **Reduce Minimum Occurrences** in sidebar")
                st.markdown("2. **Increase Max Cuts** to generate more fragments")
                st.markdown("3. **Adjust Fragment Size** constraints")
    else:
        st.error("Failed to load valid data. Please check your CSV file.")
else:
    # Show welcome message
    st.markdown("""
    ## Welcome to the MMP Analysis Tool 🧪
    
    This tool performs **Matched Molecular Pair (MMP) analysis** to identify structural transformations that affect biological activity.
    
    ### 🛠️ **Kekulization Issue Fixed**
    
    The previous version had issues with kekulization (assigning alternating single/double bonds to aromatic rings). 
    **This version handles these errors gracefully by:**
    
    1. **Skipping problematic kekulization** while keeping aromatic flags
    2. **Using alternative fragmentation methods** when RDKit's FragmentMol fails
    3. **Providing fallback options** for all chemical operations
    
    ### 📋 **How to use:**
    
    1. **Upload your CSV file** (must contain 'SMILES' and 'pIC50' columns)
    2. **Configure analysis parameters** in the sidebar
    3. **View results** including transformations and statistics
    4. **Export findings** for further analysis
    
    ### ⚙️ **Recommended settings for problematic molecules:**
    
    - ✅ **Enable "Handle kekulization errors"** (default: ON)
    - ⚠️ **Disable "Try kekulization with fallback"** for stability
    - ✅ **Use smart fragmentation** for better results
    - ⚠️ **Start with Max Cuts = 2** for reasonable performance
    
    ### 📊 **Expected CSV format:**
    ```
    SMILES,pIC50,Name
    CCOC(=O)Nc1ccc(OC)cc1,6.5,Compound1
    CC(C)NC(=O)c1ccc(O)cc1,7.2,Compound2
    ...
    ```
    
    ⬅️ **Upload a CSV file in the sidebar to get started!**
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; font-size: 0.9rem;">
<p>MMP Analysis Tool v2.1 | Kekulization Fixed | Built with Streamlit and RDKit</p>
<p>For research use only. Always validate computational predictions with experimental data.</p>
</div>
""", unsafe_allow_html=True)
