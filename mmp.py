# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from operator import itemgetter
from itertools import combinations
import sys
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

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
    .warning-box {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .cut-level {
        background-color: #E0F2FE;
        border-radius: 5px;
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-left: 3px solid #0EA5E9;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🧪 Matched Molecular Pair (MMP) Analysis Tool</h1>', unsafe_allow_html=True)

# Try to import RDKit with error handling
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.Chem.rdmolops import ReplaceCore
    RDKIT_AVAILABLE = True
except ImportError as e:
    st.error(f"RDKit not available: {e}")
    st.info("Please install RDKit with: pip install rdkit-pypi")
    RDKIT_AVAILABLE = False
except Exception as e:
    st.error(f"Error loading RDKit (NumPy compatibility issue): {e}")
    st.warning("""
    **NumPy Compatibility Issue Detected**
    
    This error occurs because RDKit was compiled with an older version of NumPy.
    
    **Solutions:**
    1. **Install specific NumPy version**: Run `pip install "numpy<2"`
    2. **Use RDKit conda package**: `conda install -c conda-forge rdkit`
    3. **Rebuild RDKit**: Recompile RDKit with NumPy 2.x support
    """)
    RDKIT_AVAILABLE = False

# Sidebar
with st.sidebar:
    st.markdown("## 📋 Configuration")
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file and RDKIT_AVAILABLE:
        # Parameters
        st.markdown("### ⚙️ Parameters")
        min_occurrence = st.slider("Minimum transform occurrences", 1, 50, 3, 
                                  help="Minimum number of occurrences for a transform to be considered")
        
        # Multi-cut fragmentation options
        st.markdown("### ✂️ Fragmentation Options")
        max_cuts = st.slider("Maximum number of cuts per molecule", 1, 5, 2,
                            help="Maximum number of bonds to cut during fragmentation")
        min_fragment_size = st.slider("Minimum fragment size (atoms)", 1, 20, 5,
                                     help="Minimum number of atoms in a fragment to consider")
        max_fragment_size = st.slider("Maximum fragment size (atoms)", 5, 100, 50,
                                     help="Maximum number of atoms in a fragment to consider")
        
        # Advanced fragmentation options
        with st.expander("Advanced Fragmentation Settings"):
            fragment_single_bonds_only = st.checkbox("Cut single bonds only", value=True,
                                                    help="Only cut single bonds during fragmentation")
            exclude_rings_from_cuts = st.checkbox("Exclude ring bonds", value=True,
                                                 help="Don't cut bonds that are part of rings")
            keep_largest_core_only = st.checkbox("Keep largest core only", value=False,
                                                help="Only keep the largest fragment as core")
            symmetric_cut_handling = st.selectbox("Symmetric cut handling", 
                                                 ["Keep all", "Remove duplicates", "Keep unique cores"],
                                                 index=0)
        
        # Molecule cleaning options
        st.markdown("### 🧹 Molecule Cleaning")
        sanitize_molecules = st.checkbox("Sanitize molecules", value=True,
                                       help="Clean molecules (recommended)")
        kekulize_molecules = st.checkbox("Kekulize molecules", value=False,
                                        help="Force kekulization (may fail for some molecules)")
        
        # Display options
        st.markdown("### 👀 Display Options")
        show_all_transforms = st.checkbox("Show all transformations", value=False)
        transforms_to_display = st.slider("Number of transforms to display", 1, 100, 20, 
                                         disabled=show_all_transforms)
        
        # Analysis options
        st.markdown("### 🔬 Analysis")
        show_top_positive = st.checkbox("Show top positive transforms", value=True)
        show_top_negative = st.checkbox("Show top negative transforms", value=True)
        show_compound_examples = st.checkbox("Show compound examples", value=True)
        show_cut_level_analysis = st.checkbox("Show cut-level analysis", value=True)
        
        # Pair generation logic
        st.markdown("### 🔗 Pair Generation Logic")
        st.info("""
        **Pairs are generated only when 3+ compounds share the same core.**
        
        This reduces noise and focuses on statistically significant transformations.
        """)
        
        # Debug option
        st.markdown("### 🐛 Debug")
        show_debug_info = st.checkbox("Show debug information", value=False)
        
        # Save options
        st.markdown("### 💾 Export")
        save_results = st.checkbox("Save results to Excel")

# Helper functions (only define if RDKit is available)
if RDKIT_AVAILABLE:
    @st.cache_data
    def load_data(file, sanitize=True, kekulize=False):
        """Load and preprocess data"""
        if file is not None:
            df = pd.read_csv(file)
            # Check required columns
            required_cols = ['SMILES', 'pIC50']
            if not all(col in df.columns for col in required_cols):
                st.error(f"CSV must contain columns: {required_cols}")
                return None
            
            # Convert SMILES to molecules with error handling
            molecules = []
            errors = []
            
            for idx, smiles in enumerate(df['SMILES']):
                try:
                    mol = Chem.MolFromSmiles(str(smiles))
                    if mol is None:
                        errors.append(f"Row {idx}: Invalid SMILES '{smiles}'")
                        molecules.append(None)
                        continue
                    
                    # Sanitize if requested
                    if sanitize:
                        try:
                            Chem.SanitizeMol(mol)
                        except:
                            pass  # Skip sanitization if it fails
                    
                    # Kekulize if requested
                    if kekulize:
                        try:
                            Chem.Kekulize(mol, clearAromaticFlags=True)
                        except:
                            pass  # Skip kekulization if it fails
                    
                    # Get largest fragment
                    frags = Chem.GetMolFrags(mol, asMols=True)
                    if frags:
                        mol = max(frags, key=lambda x: x.GetNumAtoms())
                    
                    molecules.append(mol)
                except Exception as e:
                    errors.append(f"Row {idx}: Error processing '{smiles}' - {str(e)}")
                    molecules.append(None)
            
            df['mol'] = molecules
            
            # Show errors if any
            if errors:
                with st.expander("⚠️ Processing Errors", expanded=True):
                    for error in errors[:10]:  # Show first 10 errors
                        st.warning(error)
                    if len(errors) > 10:
                        st.info(f"... and {len(errors)-10} more errors")
            
            # Remove rows with invalid molecules
            valid_df = df[df['mol'].notna()].copy()
            if len(valid_df) < len(df):
                st.warning(f"Removed {len(df) - len(valid_df)} rows with invalid molecules")
            
            return valid_df
        return None

    def remove_map_nums(mol):
        """Remove atom map numbers from a molecule"""
        if mol is None:
            return None
        for atm in mol.GetAtoms():
            atm.SetAtomMapNum(0)
        return mol

    def sort_fragments(mol, min_size=1, max_size=100):
        """Sort fragments by number of atoms, filtering by size"""
        if mol is None:
            return []
        try:
            frag_list = list(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False))
            frag_list = [remove_map_nums(x) for x in frag_list]
            
            # Filter fragments by size
            filtered_frags = []
            for frag in frag_list:
                num_atoms = frag.GetNumAtoms()
                if min_size <= num_atoms <= max_size:
                    filtered_frags.append((num_atoms, frag))
            
            # Sort by number of atoms
            filtered_frags.sort(key=itemgetter(0), reverse=True)
            return [x[1] for x in filtered_frags]
        except Exception as e:
            return []

    def find_cuttable_bonds(mol, single_bonds_only=True, exclude_rings=True):
        """Find bonds that can be cut"""
        cuttable_bonds = []
        for bond in mol.GetBonds():
            # Check bond type
            if single_bonds_only and bond.GetBondType() != Chem.BondType.SINGLE:
                continue
            
            # Check if bond is in a ring
            if exclude_rings and bond.IsInRing():
                continue
            
            # Check if cutting would create very small fragments
            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()
            
            # Simple check: don't cut bonds to terminal atoms
            if begin_atom.GetDegree() == 1 or end_atom.GetDegree() == 1:
                continue
            
            cuttable_bonds.append(bond.GetIdx())
        
        return cuttable_bonds

    def recursive_fragmentation(mol, max_cuts=2, current_cut=0, 
                               single_bonds_only=True, exclude_rings=True,
                               min_fragment_size=5, max_fragment_size=50):
        """Recursively fragment molecules with multiple cuts"""
        if current_cut >= max_cuts or mol is None:
            return []
        
        results = []
        cuttable_bonds = find_cuttable_bonds(mol, single_bonds_only, exclude_rings)
        
        for bond_idx in cuttable_bonds:
            try:
                # Create copy and cut the bond
                emol = Chem.EditableMol(mol)
                bond = mol.GetBondWithIdx(bond_idx)
                emol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                frag_mol = emol.GetMol()
                
                # Try to sanitize
                try:
                    Chem.SanitizeMol(frag_mol)
                except:
                    pass
                
                # Sort fragments
                sorted_frags = sort_fragments(frag_mol, min_fragment_size, max_fragment_size)
                
                if len(sorted_frags) >= 2:
                    # Record this fragmentation
                    frag_smiles = [Chem.MolToSmiles(f) for f in sorted_frags]
                    results.append({
                        'cut_level': current_cut + 1,
                        'bond_idx': bond_idx,
                        'fragments': sorted_frags,
                        'frag_smiles': frag_smiles,
                        'core': sorted_frags[0] if len(sorted_frags) > 0 else None,
                        'r_groups': sorted_frags[1:] if len(sorted_frags) > 1 else []
                    })
                
                # Recursively fragment further
                if current_cut + 1 < max_cuts:
                    for frag in sorted_frags:
                        sub_results = recursive_fragmentation(
                            frag, max_cuts, current_cut + 1,
                            single_bonds_only, exclude_rings,
                            min_fragment_size, max_fragment_size
                        )
                        results.extend(sub_results)
                        
            except Exception as e:
                continue
        
        return results

    def multi_cut_fragmentation(mol, max_cuts=2, **kwargs):
        """Perform multi-cut fragmentation on a molecule"""
        if mol is None:
            return []
        
        # Always include the whole molecule (0 cuts)
        results = [{
            'cut_level': 0,
            'bond_idx': None,
            'fragments': [mol],
            'frag_smiles': [Chem.MolToSmiles(mol)],
            'core': mol,
            'r_groups': []
        }]
        
        # Perform recursive fragmentation
        recursive_results = recursive_fragmentation(mol, max_cuts, **kwargs)
        results.extend(recursive_results)
        
        return results

    def get_unique_cores(fragmentation_results, keep_largest_only=False):
        """Extract unique cores from fragmentation results"""
        unique_cores = {}
        
        for result in fragmentation_results:
            if result['core'] is not None:
                try:
                    core_smiles = Chem.MolToSmiles(result['core'])
                    if core_smiles not in unique_cores:
                        unique_cores[core_smiles] = {
                            'mol': result['core'],
                            'cut_level': result['cut_level'],
                            'r_groups': result['r_groups'],
                            'frag_smiles': result['frag_smiles']
                        }
                    elif not keep_largest_only:
                        # Keep the version with more R-groups
                        current_r_count = len(unique_cores[core_smiles]['r_groups'])
                        new_r_count = len(result['r_groups'])
                        if new_r_count > current_r_count:
                            unique_cores[core_smiles] = {
                                'mol': result['core'],
                                'cut_level': result['cut_level'],
                                'r_groups': result['r_groups'],
                                'frag_smiles': result['frag_smiles']
                            }
                except:
                    continue
        
        return unique_cores

    def perform_mmp_analysis(df, min_transform_occurrence, max_cuts=2, 
                           min_fragment_size=5, max_fragment_size=50,
                           single_bonds_only=True, exclude_rings=True,
                           keep_largest_core_only=False, show_debug=False):
        """Perform MMP analysis with multi-cut support"""
        if df is None or len(df) == 0:
            return None, None, None
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Decompose molecules with multi-cut support
        status_text.text("Step 1/4: Decomposing molecules with multi-cut...")
        progress_bar.progress(25)
        
        all_rows = []
        cut_level_stats = {i: 0 for i in range(max_cuts + 1)}
        
        for idx, row in df.iterrows():
            smiles = row['SMILES']
            name = row.get('Name', f"CMPD_{idx}")
            pIC50 = row['pIC50']
            mol = row['mol']
            
            if mol is None:
                continue
                
            try:
                # Perform multi-cut fragmentation
                frag_results = multi_cut_fragmentation(
                    mol, 
                    max_cuts=max_cuts,
                    single_bonds_only=single_bonds_only,
                    exclude_rings=exclude_rings,
                    min_fragment_size=min_fragment_size,
                    max_fragment_size=max_fragment_size
                )
                
                # Get unique cores
                unique_cores = get_unique_cores(frag_results, keep_largest_core_only)
                
                # Process each unique core
                for core_smiles, core_info in unique_cores.items():
                    cut_level = core_info['cut_level']
                    cut_level_stats[cut_level] += 1
                    
                    # Process R-groups
                    for rgroup_idx, rgroup in enumerate(core_info['r_groups']):
                        try:
                            rgroup_smiles = Chem.MolToSmiles(rgroup)
                            # Add map numbers for attachment points
                            rgroup_with_star = rgroup_smiles + '[*]'
                            
                            all_rows.append([
                                smiles, core_smiles, rgroup_with_star, 
                                name, pIC50, cut_level, rgroup_idx + 1
                            ])
                        except:
                            continue
                            
            except Exception as e:
                if show_debug:
                    st.warning(f"Error fragmenting molecule {name}: {str(e)}")
                continue
        
        if not all_rows:
            st.error("No valid fragments found")
            return None, None, cut_level_stats
        
        # Create DataFrame
        row_df = pd.DataFrame(all_rows, columns=[
            "SMILES", "Core", "R_group", "Name", "pIC50", "Cut_Level", "R_Group_Num"
        ])
        
        if show_debug:
            with st.expander("Debug: Row DataFrame", expanded=False):
                st.write(f"Total rows: {len(row_df)}")
                st.write(f"Unique cores: {row_df['Core'].nunique()}")
                st.write(f"Cut level distribution: {cut_level_stats}")
                st.dataframe(row_df.head(20))
        
        # Step 2: Collect pairs
        status_text.text("Step 2/4: Collecting molecular pairs...")
        progress_bar.progress(50)
        
        delta_list = []
        core_pair_stats = {}
        
        # Group by Core and R_Group_Num (to match specific attachment points)
        for (core_smiles, r_group_num), v in row_df.groupby(["Core", "R_Group_Num"]):
            # Only process groups with more than 2 compounds
            if len(v) > 2:
                core_pair_stats[(core_smiles, r_group_num)] = len(v)
                
                # Generate all unique combinations
                for a, b in combinations(range(0, len(v)), 2):
                    reagent_a = v.iloc[a]
                    reagent_b = v.iloc[b]
                    
                    if reagent_a.SMILES == reagent_b.SMILES:
                        continue
                    
                    # Sort by SMILES for canonical ordering
                    reagent_a, reagent_b = sorted([reagent_a, reagent_b], key=lambda x: x.SMILES)
                    
                    # Calculate delta
                    delta = reagent_b.pIC50 - reagent_a.pIC50
                    
                    # Create transform string
                    transform_str = f"{reagent_a.R_group.replace('[*]','*-')}>>{reagent_b.R_group.replace('[*]','*-')}"
                    
                    delta_list.append([
                        reagent_a.SMILES, reagent_a.Core, reagent_a.R_group, reagent_a.Name, reagent_a.pIC50, reagent_a.Cut_Level,
                        reagent_b.SMILES, reagent_b.Core, reagent_b.R_group, reagent_b.Name, reagent_b.pIC50, reagent_b.Cut_Level,
                        transform_str, delta
                    ])
        
        if not delta_list:
            st.error("No molecular pairs found")
            return None, None, cut_level_stats
        
        # Create DataFrame
        cols = [
            "SMILES_1", "Core_1", "R_group_1", "Name_1", "pIC50_1", "Cut_Level_1",
            "SMILES_2", "Core_2", "R_group_2", "Name_2", "pIC50_2", "Cut_Level_2",
            "Transform", "Delta"
        ]
        delta_df = pd.DataFrame(delta_list, columns=cols)
        
        if show_debug:
            with st.expander("Debug: Delta DataFrame", expanded=False):
                st.write(f"Delta DataFrame shape: {delta_df.shape}")
                st.write(f"Cores with pairs: {len(core_pair_stats)}")
                st.dataframe(delta_df.head(10))
        
        # Step 3: Collect frequent transforms
        status_text.text("Step 3/4: Analyzing transformations...")
        progress_bar.progress(75)
        
        mmp_list = []
        for k, v in delta_df.groupby("Transform"):
            # Only include transforms with minimum occurrences
            if len(v) >= min_transform_occurrence:
                mmp_list.append([
                    k, len(v), v.Delta.values,
                    v['Cut_Level_1'].iloc[0],  # Cut level for this transform
                    list(v['Core_1'].unique())  # Cores where this transform appears
                ])
        
        if not mmp_list:
            st.warning(f"No transforms found with {min_transform_occurrence}+ occurrences")
            return delta_df, None, cut_level_stats
        
        # Create transforms DataFrame
        mmp_df = pd.DataFrame(mmp_list, columns=[
            "Transform", "Count", "Deltas", "Cut_Level", "Cores"
        ])
        mmp_df['mean_delta'] = [x.mean() for x in mmp_df.Deltas]
        
        # Create reaction molecules
        rxn_mols = []
        for transform in mmp_df['Transform']:
            try:
                rxn = AllChem.ReactionFromSmarts(transform.replace('*-','*'), useSmiles=True)
                rxn_mols.append(rxn)
            except:
                try:
                    parts = transform.split('>>')
                    if len(parts) == 2:
                        left = parts[0].replace('*-','*')
                        right = parts[1].replace('*-','*')
                        rxn_smarts = f"{left}>>{right}"
                        rxn = AllChem.ReactionFromSmarts(rxn_smarts, useSmiles=True)
                        rxn_mols.append(rxn)
                    else:
                        rxn_mols.append(None)
                except:
                    rxn_mols.append(None)
        
        mmp_df['rxn_mol'] = rxn_mols
        
        # Step 4: Complete
        status_text.text("Step 4/4: Analysis complete!")
        progress_bar.progress(100)
        status_text.empty()
        progress_bar.empty()
        
        return delta_df, mmp_df, cut_level_stats

    def plot_stripplot_to_fig(deltas):
        """Create a stripplot figure"""
        fig, ax = plt.subplots(figsize=(4, 1.5))
        if len(deltas) > 0:
            sns.stripplot(x=deltas, ax=ax, jitter=0.2, alpha=0.7, s=5, color='blue')
            ax.axvline(0, ls='--', c='red')
            
            data_min = min(deltas)
            data_max = max(deltas)
            padding = max(0.5, (data_max - data_min) * 0.1)
            ax.set_xlim(data_min - padding, data_max + padding)
        else:
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
        try:
            img = Draw.ReactionToImage(rxn_mol, subImgSize=(300, 150))
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode()
        except:
            return None

    def find_examples(delta_df, transform):
        """Find examples for a specific transform"""
        examples = delta_df[delta_df['Transform'] == transform]
        if len(examples) == 0:
            return None
        
        example_list = []
        for _, row in examples.sort_values("Delta", ascending=False).iterrows():
            example_list.append({
                "SMILES": row['SMILES_1'],
                "Name": row['Name_1'],
                "pIC50": row['pIC50_1'],
                "Type": "Before",
                "Cut_Level": row['Cut_Level_1']
            })
            example_list.append({
                "SMILES": row['SMILES_2'],
                "Name": row['Name_2'],
                "pIC50": row['pIC50_2'],
                "Type": "After",
                "Cut_Level": row['Cut_Level_2']
            })
        
        return pd.DataFrame(example_list)

    def visualize_fragmentation_example(mol, max_cuts=2, **kwargs):
        """Visualize fragmentation for a molecule"""
        if mol is None:
            return None
        
        frag_results = multi_cut_fragmentation(mol, max_cuts=max_cuts, **kwargs)
        
        visualization_data = []
        for result in frag_results:
            if result['cut_level'] > 0 and len(result['fragments']) >= 2:
                try:
                    # Create a simple visualization
                    core_img = Draw.MolToImage(result['core'], size=(200, 200))
                    rgroup_imgs = [Draw.MolToImage(rg, size=(100, 100)) for rg in result['r_groups']]
                    
                    visualization_data.append({
                        'cut_level': result['cut_level'],
                        'core': result['core'],
                        'core_smiles': Chem.MolToSmiles(result['core']),
                        'r_groups': result['r_groups'],
                        'rgroup_smiles': [Chem.MolToSmiles(rg) for rg in result['r_groups']]
                    })
                except:
                    continue
        
        return visualization_data

# Main app logic
if not RDKIT_AVAILABLE:
    st.error("""
    ## RDKit Not Available
    
    This app requires RDKit for chemical informatics functionality.
    
    **To install RDKit:**
    
    ### Option 1: Using pip (may have NumPy compatibility issues)
    ```bash
    pip install rdkit-pypi
    pip install "numpy<2"  # Downgrade NumPy for compatibility
    ```
    
    ### Option 2: Using conda (recommended)
    ```bash
    conda install -c conda-forge rdkit
    ```
    
    ### Option 3: Manual installation
    1. Visit: https://www.rdkit.org/docs/Install.html
    2. Follow installation instructions for your platform
    
    **Note:** If you see NumPy compatibility errors, you need to downgrade NumPy:
    ```bash
    pip install "numpy<2"
    ```
    """)
    
elif uploaded_file is not None:
    # Get parameters from sidebar
    sanitize = sanitize_molecules
    kekulize = kekulize_molecules
    show_debug = show_debug_info if 'show_debug_info' in locals() else False
    
    # Load data
    df = load_data(uploaded_file, sanitize=sanitize, kekulize=kekulize)
    
    if df is not None and len(df) > 0:
        # Show dataset info
        with st.expander("📊 Dataset Overview", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Compounds", len(df))
            col2.metric("Min pIC50", f"{df['pIC50'].min():.2f}")
            col3.metric("Max pIC50", f"{df['pIC50'].max():.2f}")
            col4.metric("Avg pIC50", f"{df['pIC50'].mean():.2f}")
            
            # Show sample molecules
            st.subheader("Sample Molecules")
            cols = st.columns(4)
            for idx, (_, row) in enumerate(df.head(8).iterrows()):
                with cols[idx % 4]:
                    try:
                        mol = row['mol']
                        if mol:
                            img = Draw.MolToImage(mol, size=(200, 200))
                            name = row.get('Name', f"Compound {idx+1}")
                            st.image(img, caption=f"{name} (pIC50: {row['pIC50']:.2f})")
                    except:
                        pass
        
        # Perform MMP analysis with multi-cut
        st.markdown('<h2 class="section-header">🔍 MMP Analysis with Multi-Cut Support</h2>', unsafe_allow_html=True)
        
        delta_df, mmp_df, cut_level_stats = perform_mmp_analysis(
            df, 
            min_occurrence,
            max_cuts=max_cuts,
            min_fragment_size=min_fragment_size,
            max_fragment_size=max_fragment_size,
            single_bonds_only=fragment_single_bonds_only,
            exclude_rings=exclude_rings_from_cuts,
            keep_largest_core_only=keep_largest_core_only,
            show_debug=show_debug
        )
        
        if delta_df is not None:
            # Show cut-level statistics
            if show_cut_level_analysis and cut_level_stats:
                st.markdown('<h3 class="section-header">✂️ Cut Level Analysis</h3>', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("0-cut (Whole molecule)", cut_level_stats.get(0, 0))
                with col2:
                    st.metric("1-cut pairs", cut_level_stats.get(1, 0))
                with col3:
                    total_multi_cut = sum(v for k, v in cut_level_stats.items() if k > 1)
                    st.metric("Multi-cut (2+) pairs", total_multi_cut)
                
                # Show distribution
                fig, ax = plt.subplots(figsize=(8, 4))
                levels = list(cut_level_stats.keys())
                counts = [cut_level_stats[l] for l in levels]
                
                bars = ax.bar(levels, counts, color='skyblue', edgecolor='black')
                ax.set_xlabel('Cut Level')
                ax.set_ylabel('Number of Fragments')
                ax.set_title('Fragmentation by Cut Level')
                ax.set_xticks(levels)
                
                # Add count labels on bars
                for bar, count in zip(bars, counts):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                           f'{count}', ha='center', va='bottom')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Explanation of cut levels
                with st.expander("📝 What do cut levels mean?"):
                    st.markdown("""
                    **Cut Levels Explained:**
                    - **0-cut**: Whole molecule treated as core (no fragmentation)
                    - **1-cut**: Single bond cut, creating 2 fragments
                    - **2-cut**: Two bonds cut, potentially creating 3 fragments
                    - **3+ cuts**: Complex fragmentation patterns
                    
                    **Multi-cut advantages:**
                    1. Identifies more complex transformations
                    2. Can handle multiple R-group modifications simultaneously
                    3. Better for analyzing complex scaffolds
                    4. More comprehensive coverage of chemical space
                    """)
            
            # Show general statistics
            st.success(f"Analysis complete! Found {len(delta_df)} molecular pairs")
            
            if mmp_df is not None:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Pairs", len(delta_df))
                col2.metric("Unique Transforms", len(mmp_df))
                col3.metric("Avg Occurrences", f"{mmp_df['Count'].mean():.1f}")
                col4.metric("Max Cut Level", f"{mmp_df['Cut_Level'].max():.0f}")
                
                # Sort transforms by mean delta
                mmp_df_sorted = mmp_df.sort_values("mean_delta", ascending=False)
                
                # Group by cut level for analysis
                mmp_df_by_cut = mmp_df_sorted.groupby('Cut_Level')
                
                # Show transforms by cut level
                if show_cut_level_analysis:
                    for cut_level, group in mmp_df_by_cut:
                        if len(group) > 0:
                            st.markdown(f'<div class="cut-level"><h4>Cut Level {cut_level} Transforms</h4></div>', unsafe_allow_html=True)
                            
                            # Show top 3 from this cut level
                            top_transforms = group.head(3)
                            
                            for i, (_, row) in enumerate(top_transforms.iterrows()):
                                with st.container():
                                    col1, col2, col3 = st.columns([1, 2, 1])
                                    
                                    with col1:
                                        img_base64 = get_rxn_image(row['rxn_mol'])
                                        if img_base64:
                                            st.markdown(f'<img src="data:image/png;base64,{img_base64}" width="300">', unsafe_allow_html=True)
                                        else:
                                            st.info("Image not available")
                                    
                                    with col2:
                                        st.markdown(f"""
                                        <div class="transform-card">
                                            <h4>Transform (Cut Level {cut_level})</h4>
                                            <p><strong>Transformation:</strong> {row['Transform']}</p>
                                            <p><strong>Mean ΔpIC50:</strong> {row['mean_delta']:.2f}</p>
                                            <p><strong>Occurrences:</strong> {row['Count']}</p>
                                            <p><strong>Cores Found:</strong> {len(row['Cores'])} unique cores</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    
                                    with col3:
                                        fig = plot_stripplot_to_fig(row['Deltas'])
                                        st.pyplot(fig)
                
                # Show top positive transforms (all cut levels combined)
                if show_top_positive and len(mmp_df_sorted) > 0:
                    st.markdown('<h3 class="section-header">📈 Top Positive Transformations (All Cut Levels)</h3>', unsafe_allow_html=True)
                    top_positive = mmp_df_sorted.head(min(5, len(mmp_df_sorted)))
                    
                    for i, (_, row) in enumerate(top_positive.iterrows()):
                        with st.container():
                            col1, col2, col3 = st.columns([1, 2, 1])
                            
                            with col1:
                                img_base64 = get_rxn_image(row['rxn_mol'])
                                if img_base64:
                                    st.markdown(f'<img src="data:image/png;base64,{img_base64}" width="300">', unsafe_allow_html=True)
                                else:
                                    st.info("Reaction image not available")
                            
                            with col2:
                                st.markdown(f"""
                                <div class="transform-card">
                                    <h4>Top Transform #{i+1} (Cut Level {row['Cut_Level']})</h4>
                                    <p><strong>Transformation:</strong> {row['Transform']}</p>
                                    <p><strong>Mean ΔpIC50:</strong> {row['mean_delta']:.2f}</p>
                                    <p><strong>Occurrences:</strong> {row['Count']}</p>
                                    <p><strong>ΔpIC50 Range:</strong> {min(row['Deltas']):.2f} to {max(row['Deltas']):.2f}</p>
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
                                        st.dataframe(examples_df)
                
                # Show top negative transforms
                if show_top_negative and len(mmp_df_sorted) > 0:
                    st.markdown('<h3 class="section-header">📉 Top Negative Transformations</h3>', unsafe_allow_html=True)
                    top_negative = mmp_df_sorted.tail(min(5, len(mmp_df_sorted))).iloc[::-1]
                    
                    for i, (_, row) in enumerate(top_negative.iterrows()):
                        with st.container():
                            col1, col2, col3 = st.columns([1, 2, 1])
                            
                            with col1:
                                img_base64 = get_rxn_image(row['rxn_mol'])
                                if img_base64:
                                    st.markdown(f'<img src="data:image/png;base64,{img_base64}" width="300">', unsafe_allow_html=True)
                                else:
                                    st.info("Reaction image not available")
                            
                            with col2:
                                st.markdown(f"""
                                <div class="transform-card">
                                    <h4>Top Negative Transform #{i+1} (Cut Level {row['Cut_Level']})</h4>
                                    <p><strong>Transformation:</strong> {row['Transform']}</p>
                                    <p><strong>Mean ΔpIC50:</strong> {row['mean_delta']:.2f}</p>
                                    <p><strong>Occurrences:</strong> {row['Count']}</p>
                                    <p><strong>ΔpIC50 Range:</strong> {min(row['Deltas']):.2f} to {max(row['Deltas']):.2f}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col3:
                                fig = plot_stripplot_to_fig(row['Deltas'])
                                st.pyplot(fig)
                
                # Show all transforms table
                if len(mmp_df_sorted) > 0:
                    st.markdown('<h3 class="section-header">📋 All Transformations</h3>', unsafe_allow_html=True)
                    
                    if show_all_transforms:
                        display_df = mmp_df_sorted
                    else:
                        display_df = mmp_df_sorted.head(transforms_to_display)
                    
                    # Display with cut level information
                    display_columns = ['Transform', 'Cut_Level', 'Count', 'mean_delta']
                    st.dataframe(display_df[display_columns].rename(
                        columns={
                            'mean_delta': 'Mean ΔpIC50',
                            'Cut_Level': 'Cuts'
                        }
                    ).round(3))
                    
                    # Option to view full data
                    with st.expander("View detailed transform data"):
                        st.dataframe(display_df)
                
                # Export results
                if save_results and mmp_df_sorted is not None:
                    st.markdown('<h3 class="section-header">💾 Export Results</h3>', unsafe_allow_html=True)
                    
                    @st.cache_data
                    def convert_df_to_csv(df):
                        return df.to_csv(index=False).encode('utf-8')
                    
                    @st.cache_data
                    def convert_df_to_excel(df_dict):
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            for sheet_name, df in df_dict.items():
                                df.to_excel(writer, index=False, sheet_name=sheet_name)
                        return output.getvalue()
                    
                    # Prepare data for export
                    export_data = {
                        'MMP_Results': mmp_df_sorted,
                        'Molecular_Pairs': delta_df,
                        'Cut_Level_Stats': pd.DataFrame({
                            'Cut_Level': list(cut_level_stats.keys()),
                            'Count': list(cut_level_stats.values())
                        })
                    }
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.download_button(
                            label="📥 Download MMP Results (CSV)",
                            data=convert_df_to_csv(mmp_df_sorted),
                            file_name="mmp_results_multi_cut.csv",
                            mime="text/csv"
                        )
                    
                    with col2:
                        excel_data = convert_df_to_excel(export_data)
                        st.download_button(
                            label="📥 Download Complete Analysis (Excel)",
                            data=excel_data,
                            file_name="mmp_analysis_multi_cut.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
            
            else:
                st.info(f"No transformations found with {min_occurrence}+ occurrences. Try reducing the minimum occurrence threshold or increasing max cuts.")
    else:
        st.warning("No valid molecules found in the dataset. Please check your SMILES strings.")
else:
    # Show welcome message
    st.markdown("""
    ## Welcome to the MMP Analysis Tool with Multi-Cut Support! 👋
    
    This enhanced tool performs **Matched Molecular Pair (MMP) analysis with multi-cut fragmentation** to identify complex structural transformations.
    
    ### New Multi-Cut Features:
    1. **Multiple bond cuts**: Cut up to 5 bonds per molecule
    2. **Complex fragmentation**: Identify transformations involving multiple R-groups
    3. **Cut-level analysis**: Analyze transformations by fragmentation complexity
    4. **Size filtering**: Control fragment sizes for meaningful analysis
    
    ### Key Improvements:
    - **Better for complex molecules**: Can handle multi-point modifications
    - **More comprehensive**: Identifies transformations missed by single-cut methods
    - **Flexible analysis**: Control cut parameters for your specific needs
    
    ### How to use:
    1. **Upload your data** using the sidebar
    2. **Configure fragmentation parameters** (max cuts, fragment sizes)
    3. **Set analysis parameters** (minimum occurrences, etc.)
    4. **View results** including cut-level analysis
    5. **Export findings** for further analysis
    
    ### Example Use Cases:
    - **SAR analysis** of compounds with multiple substitution points
    - **Scaffold hopping** with complex transformations
    - **Multi-parameter optimization** studies
    - **Fragment-based drug design**
    
    ⬅️ **Upload a CSV file in the sidebar to get started!**
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; font-size: 0.9rem;">
    <p>MMP Analysis Tool v2.0 | Multi-Cut Support | Built with Streamlit and RDKit</p>
    <p>For research use only. Always validate computational predictions with experimental data.</p>
</div>
""", unsafe_allow_html=True)
