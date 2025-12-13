# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import warnings
from collections import defaultdict
import itertools
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Advanced MMP Analysis Tool (Single Cut)",
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
        background-color: #DCFCE7;
        border-left: 4px solid #16A34A;
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
    .highlight-before {
        background-color: #FFE4E4;
        padding: 2px 4px;
        border-radius: 3px;
        font-weight: bold;
        color: #DC2626;
    }
    .highlight-after {
        background-color: #DCFCE7;
        padding: 2px 4px;
        border-radius: 3px;
        font-weight: bold;
        color: #16A34A;
    }
    .stProgress > div > div > div > div {
        background-color: #3B82F6;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🧪 Advanced MMP Analysis Tool (Single Cut)</h1>', unsafe_allow_html=True)

# Try to import RDKit with error handling
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw, rdFMCS
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.Chem import rdDepictor, Descriptors, rdMolDescriptors
    from rdkit.Chem import rdMMPA
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDKIT_AVAILABLE = True
except ImportError as e:
    st.error(f"RDKit not available: {e}")
    st.info("Please install RDKit with: pip install rdkit-pypi")
    RDKIT_AVAILABLE = False
except Exception as e:
    st.error(f"Error loading RDKit: {e}")
    RDKIT_AVAILABLE = False

# Sidebar configuration
with st.sidebar:
    st.markdown("## 📋 Configuration")
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file and RDKIT_AVAILABLE:
        st.markdown("### ⚙️ Analysis Parameters")
        
        # MMPA method selection
        st.markdown("#### 🔗 MMPA Method")
        mmpa_method = st.selectbox(
            "Select MMPA method",
            ["Standard Single Cut", "Side-chain Only", "Exhaustive Single Cut"],
            help="Standard: Single non-ring cuts\nSide-chain: Only terminal cuts\nExhaustive: All single cuts including rings"
        )
        
        # Minimum requirements
        st.markdown("#### 📊 Minimum Requirements")
        min_pairs_per_core = st.slider("Minimum compounds per core", 2, 10, 2,
                                      help="Minimum number of compounds sharing the same core")
        min_transform_occurrence = st.slider("Minimum transform occurrences", 1, 20, 1,
                                           help="Minimum occurrences for statistical significance")
        
        # Fragment filters
        st.markdown("#### 🪓 Fragment Filters")
        min_core_atoms = st.slider("Minimum core atoms", 5, 50, 10,
                                  help="Minimum number of atoms in core fragment")
        max_rgroup_atoms = st.slider("Maximum R-group atoms", 5, 50, 20,
                                   help="Maximum number of atoms in R-group")
        
        # Property filters
        st.markdown("#### 🔍 Property Filters")
        min_mw = st.number_input("Minimum MW", 0.0, 1000.0, 100.0, 10.0,
                                help="Minimum molecular weight")
        max_mw = st.number_input("Maximum MW", 100.0, 2000.0, 500.0, 10.0,
                                help="Maximum molecular weight")
        
        # Display options
        st.markdown("### 👀 Display Options")
        n_top_transforms = st.slider("Top transforms to display", 1, 50, 10)
        show_fragment_images = st.checkbox("Show fragment images", value=True)
        show_detailed_debug = st.checkbox("Show detailed debug info", value=False)
        
        # Export options
        st.markdown("### 💾 Export")
        export_all_data = st.checkbox("Export all data", value=True)
        
        # About
        with st.expander("ℹ️ About This Tool"):
            st.markdown("""
            **Advanced MMP Analysis Tool (Single Cut)**
            
            This tool uses RDKit's MMPA for single-cut analysis:
            
            1. **Standard Single Cut**: Single non-ring bond cuts only
            2. **Side-chain Only**: Only terminal bond cuts
            3. **Exhaustive Single Cut**: All single bond cuts including rings
            
            **Key Features:**
            - Uses RDKit's rdMMPA.FragmentMol for robust fragmentation
            - Filters fragments by size and properties
            - Generates statistically significant transforms
            
            **Tips for success:**
            - Start with larger datasets (>20 compounds)
            - Adjust minimum core size to match your scaffolds
            - Check fragment images to validate cuts
            """)

# Helper functions
if RDKIT_AVAILABLE:
    @st.cache_data
    def load_and_preprocess_data(file):
        """Load and preprocess CSV data with comprehensive validation"""
        if file is None:
            return None
        
        try:
            df = pd.read_csv(file)
            
            # Validate required columns
            required_cols = ['SMILES']
            if not all(col in df.columns for col in required_cols):
                st.error(f"CSV must contain: {required_cols}")
                return None
            
            # Add pIC50 if not present (for testing)
            if 'pIC50' not in df.columns:
                st.warning("pIC50 column not found. Using random values for demonstration.")
                np.random.seed(42)
                df['pIC50'] = np.random.uniform(4.0, 8.0, len(df))
            
            # Add Name if not present
            if 'Name' not in df.columns:
                df['Name'] = [f"Compound_{i+1}" for i in range(len(df))]
            
            # Clean SMILES
            df['SMILES'] = df['SMILES'].astype(str).str.strip()
            
            # Convert SMILES to molecules
            molecules = []
            valid_indices = []
            
            for idx, row in df.iterrows():
                try:
                    mol = Chem.MolFromSmiles(row['SMILES'])
                    if mol is not None:
                        # Basic sanitization
                        Chem.SanitizeMol(mol)
                        # Remove salts and keep largest fragment
                        frags = Chem.GetMolFrags(mol, asMols=True)
                        if frags:
                            mol = max(frags, key=lambda x: x.GetNumAtoms())
                        
                        # Check molecular weight
                        mw = Descriptors.MolWt(mol)
                        if min_mw <= mw <= max_mw:
                            molecules.append(mol)
                            valid_indices.append(idx)
                        else:
                            st.warning(f"Compound {row.get('Name', idx)} MW {mw:.1f} outside range")
                    else:
                        st.warning(f"Invalid SMILES at row {idx}: {row['SMILES']}")
                except Exception as e:
                    st.warning(f"Error processing row {idx}: {e}")
            
            if not molecules:
                st.error("No valid molecules found after preprocessing")
                return None
            
            # Create final dataframe
            final_df = df.iloc[valid_indices].copy()
            final_df['mol'] = molecules
            
            # Add molecular properties
            final_df['MW'] = [Descriptors.MolWt(mol) for mol in molecules]
            final_df['LogP'] = [Descriptors.MolLogP(mol) for mol in molecules]
            final_df['HBA'] = [Descriptors.NumHAcceptors(mol) for mol in molecules]
            final_df['HBD'] = [Descriptors.NumHDonors(mol) for mol in molecules]
            final_df['TPSA'] = [Descriptors.TPSA(mol) for mol in molecules]
            final_df['RotatableBonds'] = [Descriptors.NumRotatableBonds(mol) for mol in molecules]
            
            st.success(f"Loaded {len(final_df)} valid compounds")
            return final_df
            
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None
    
    def generate_fragments_single_cut(mol, method="standard"):
        """Generate fragments using RDKit MMPA with single cuts only"""
        fragments = []
        
        try:
            if method == "Side-chain Only":
                # Only cut terminal bonds (bonds where one atom has degree 1)
                for bond in mol.GetBonds():
                    a1 = bond.GetBeginAtom()
                    a2 = bond.GetEndAtom()
                    if (a1.GetDegree() == 1 or a2.GetDegree() == 1) and not bond.IsInRing():
                        try:
                            emol = Chem.EditableMol(mol)
                            emol.RemoveBond(a1.GetIdx(), a2.GetIdx())
                            frag_mol = emol.GetMol()
                            
                            frags = Chem.GetMolFrags(frag_mol, asMols=True, sanitizeFrags=False)
                            if len(frags) == 2:
                                # Sort by size (largest first as core)
                                frags_sorted = sorted(frags, key=lambda x: x.GetNumAtoms(), reverse=True)
                                
                                core_mol = frags_sorted[0]
                                rgroup_mol = frags_sorted[1]
                                
                                # Check size constraints
                                if (core_mol.GetNumAtoms() >= min_core_atoms and 
                                    rgroup_mol.GetNumAtoms() <= max_rgroup_atoms):
                                    
                                    # Add attachment points
                                    core_smiles = Chem.MolToSmiles(core_mol) + '[*]'
                                    rgroup_smiles = Chem.MolToSmiles(rgroup_mol) + '[*]'
                                    
                                    fragments.append({
                                        'core_mol': core_mol,
                                        'rgroup_mol': rgroup_mol,
                                        'core_smiles': core_smiles,
                                        'rgroup_smiles': rgroup_smiles,
                                        'core_size': core_mol.GetNumAtoms(),
                                        'rgroup_size': rgroup_mol.GetNumAtoms(),
                                        'bond_type': str(bond.GetBondType()),
                                        'is_ring': bond.IsInRing(),
                                        'is_terminal': (a1.GetDegree() == 1 or a2.GetDegree() == 1)
                                    })
                        except:
                            continue
            
            else:
                # Use RDKit's FragmentMol for systematic single bond fragmentation
                max_cuts = 1  # Single cuts only
                
                if method == "Exhaustive Single Cut":
                    # Allow all single bond cuts including rings
                    results = rdMMPA.FragmentMol(
                        mol,
                        maxCuts=max_cuts,
                        maxCutBonds=100,  # High number for exhaustive
                        pattern="[*:1]~[*:2]",
                        resultsAsMols=False
                    )
                else:  # Standard Single Cut
                    # More conservative settings
                    results = rdMMPA.FragmentMol(
                        mol,
                        maxCuts=max_cuts,
                        maxCutBonds=50,
                        pattern="[*:1]-[*:2]",  # Only single bonds
                        resultsAsMols=False
                    )
                
                for core_smiles, rgroup_smiles in results:
                    try:
                        # Clean the SMILES
                        core_smiles = core_smiles.replace('[*:1]', '*').replace('[*:2]', '*')
                        rgroup_smiles = rgroup_smiles.replace('[*:1]', '*').replace('[*:2]', '*')
                        
                        # Convert back to molecules for validation
                        core_mol = Chem.MolFromSmiles(core_smiles)
                        rgroup_mol = Chem.MolFromSmiles(rgroup_smiles)
                        
                        if core_mol and rgroup_mol:
                            # Check size constraints
                            if (core_mol.GetNumAtoms() >= min_core_atoms and 
                                rgroup_mol.GetNumAtoms() <= max_rgroup_atoms):
                                
                                # Standardize attachment points
                                core_smiles_clean = core_smiles.replace('[*:1]', '*').replace('[*:2]', '*')
                                rgroup_smiles_clean = rgroup_smiles.replace('[*:1]', '*').replace('[*:2]', '*')
                                
                                fragments.append({
                                    'core_mol': core_mol,
                                    'rgroup_mol': rgroup_mol,
                                    'core_smiles': core_smiles_clean,
                                    'rgroup_smiles': rgroup_smiles_clean,
                                    'core_size': core_mol.GetNumAtoms(),
                                    'rgroup_size': rgroup_mol.GetNumAtoms(),
                                    'bond_type': 'Single',
                                    'is_ring': False,  # RDKit handles ring vs non-ring
                                    'is_terminal': False
                                })
                    except:
                        continue
        
        except Exception as e:
            st.warning(f"Error in fragmentation: {e}")
        
        return fragments
    
    def visualize_fragments(compound_name, mol, fragments):
        """Visualize fragmentation results"""
        if not fragments or not show_fragment_images:
            return
        
        st.markdown(f"**Fragmentation for {compound_name}**")
        
        # Create a grid of images
        n_frags = min(len(fragments), 6)  # Show max 6 fragments
        cols = st.columns(3)
        
        for idx in range(0, n_frags, 3):
            for col_idx, col in enumerate(cols):
                frag_idx = idx + col_idx
                if frag_idx < n_frags:
                    frag = fragments[frag_idx]
                    
                    # Create combined visualization
                    try:
                        # Create molecule with attachment point highlighted
                        core_with_attach = Chem.MolFromSmiles(frag['core_smiles'].replace('[*]', '[#0]'))
                        rgroup_with_attach = Chem.MolFromSmiles(frag['rgroup_smiles'].replace('[*]', '[#0]'))
                        
                        if core_with_attach and rgroup_with_attach:
                            # Highlight attachment point
                            for atom in core_with_attach.GetAtoms():
                                if atom.GetAtomicNum() == 0:
                                    atom.SetProp("atomNote", "Attachment")
                            
                            img = Draw.MolsToGridImage(
                                [core_with_attach, rgroup_with_attach],
                                molsPerRow=2,
                                subImgSize=(200, 150),
                                legends=[f"Core ({frag['core_size']} atoms)", 
                                        f"R-group ({frag['rgroup_size']} atoms)"]
                            )
                            
                            col.image(img, use_container_width=True)
                            col.caption(f"SMILES: {frag['core_smiles'][:30]}...")
                    except:
                        pass
    
    def perform_mmp_analysis_single_cut(df, method, min_pairs_per_core, show_debug=False):
        """Perform MMP analysis with single cuts only"""
        
        st.info(f"Starting MMP analysis using {method} method...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Generate fragments for all compounds
        status_text.text("Step 1/4: Generating fragments...")
        progress_bar.progress(25)
        
        compound_fragments = {}
        all_fragments = []
        
        for idx, row in df.iterrows():
            mol = row['mol']
            fragments = generate_fragments_single_cut(mol, method)
            
            compound_fragments[idx] = {
                'name': row['Name'],
                'smiles': row['SMILES'],
                'pIC50': row['pIC50'],
                'fragments': fragments,
                'mol': mol
            }
            
            all_fragments.extend(fragments)
            
            # Show fragment visualization for first few compounds
            if idx < 3 and show_fragment_images:
                visualize_fragments(row['Name'], mol, fragments)
        
        if not all_fragments:
            st.error(f"No fragments generated with {method} method. Try:")
            st.markdown("""
            1. Reduce minimum core atoms (currently {min_core_atoms})
            2. Increase maximum R-group atoms (currently {max_rgroup_atoms})
            3. Try 'Exhaustive Single Cut' method
            4. Check if molecules have cuttable bonds
            """.format(min_core_atoms=min_core_atoms, max_rgroup_atoms=max_rgroup_atoms))
            return None, None
        
        # Debug: Show fragment statistics
        if show_debug:
            with st.expander("📊 Fragment Statistics", expanded=False):
                frag_df = pd.DataFrame(all_fragments)
                st.write(f"Total fragments generated: {len(all_fragments)}")
                st.write(f"Average fragments per compound: {len(all_fragments)/len(df):.1f}")
                
                if len(frag_df) > 0:
                    st.write(f"Core size range: {frag_df['core_size'].min()} - {frag_df['core_size'].max()} atoms")
                    st.write(f"R-group size range: {frag_df['rgroup_size'].min()} - {frag_df['rgroup_size'].max()} atoms")
                    
                    # Plot size distributions
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
                    
                    ax1.hist(frag_df['core_size'], bins=20, alpha=0.7, color='blue')
                    ax1.set_xlabel('Core atoms')
                    ax1.set_ylabel('Frequency')
                    ax1.set_title('Core Size Distribution')
                    
                    ax2.hist(frag_df['rgroup_size'], bins=20, alpha=0.7, color='green')
                    ax2.set_xlabel('R-group atoms')
                    ax2.set_ylabel('Frequency')
                    ax2.set_title('R-group Size Distribution')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
        
        # Step 2: Group compounds by common cores
        status_text.text("Step 2/4: Grouping by common cores...")
        progress_bar.progress(50)
        
        core_to_compounds = defaultdict(list)
        
        for comp_idx, comp_data in compound_fragments.items():
            seen_cores = set()
            for frag in comp_data['fragments']:
                core_smiles = frag['core_smiles']
                
                # Avoid adding same compound multiple times for same core
                if core_smiles not in seen_cores:
                    core_to_compounds[core_smiles].append({
                        'comp_idx': comp_idx,
                        'name': comp_data['name'],
                        'smiles': comp_data['smiles'],
                        'pIC50': comp_data['pIC50'],
                        'rgroup_smiles': frag['rgroup_smiles'],
                        'core_mol': frag['core_mol'],
                        'rgroup_mol': frag['rgroup_mol']
                    })
                    seen_cores.add(core_smiles)
        
        # Filter groups by minimum size
        valid_groups = {core: comps for core, comps in core_to_compounds.items() 
                       if len(comps) >= min_pairs_per_core}
        
        if not valid_groups:
            st.warning(f"No cores found with {min_pairs_per_core}+ compounds. Try:")
            st.markdown(f"""
            1. Reduce minimum compounds per core (currently {min_pairs_per_core})
            2. Reduce minimum core atoms (currently {min_core_atoms})
            3. Check if fragments are being generated correctly
            """)
            return None, None
        
        # Debug: Show group statistics
        if show_debug:
            with st.expander("📊 Group Statistics", expanded=False):
                st.write(f"Total unique cores: {len(core_to_compounds)}")
                st.write(f"Valid cores (≥{min_pairs_per_core} compounds): {len(valid_groups)}")
                
                group_sizes = [len(comps) for comps in valid_groups.values()]
                st.write(f"Average group size: {np.mean(group_sizes):.1f}")
                st.write(f"Largest group: {max(group_sizes)} compounds")
                
                # Show top 10 cores
                st.subheader("Top 10 largest cores:")
                top_cores = sorted(valid_groups.items(), key=lambda x: len(x[1]), reverse=True)[:10]
                for core, comps in top_cores:
                    st.write(f"Core SMILES: {core[:60]}...")
                    st.write(f"  Compounds: {len(comps)} | Core size: {comps[0]['core_mol'].GetNumAtoms()} atoms")
        
        # Step 3: Generate pairs
        status_text.text("Step 3/4: Generating molecular pairs...")
        progress_bar.progress(75)
        
        all_pairs = []
        
        for core, compounds in valid_groups.items():
            # Generate all unique pairs
            for i in range(len(compounds)):
                for j in range(i + 1, len(compounds)):
                    comp1 = compounds[i]
                    comp2 = compounds[j]
                    
                    # Skip if same compound
                    if comp1['comp_idx'] == comp2['comp_idx']:
                        continue
                    
                    # Calculate delta pIC50
                    delta = comp2['pIC50'] - comp1['pIC50']
                    
                    # Create transform string
                    transform = f"{comp1['rgroup_smiles']}>>{comp2['rgroup_smiles']}"
                    
                    # Store pair
                    all_pairs.append({
                        'Core_SMILES': core,
                        'Core_Atoms': comp1['core_mol'].GetNumAtoms(),
                        'Compound1_Name': comp1['name'],
                        'Compound1_SMILES': comp1['smiles'],
                        'Compound1_pIC50': comp1['pIC50'],
                        'Compound1_Rgroup': comp1['rgroup_smiles'],
                        'Compound2_Name': comp2['name'],
                        'Compound2_SMILES': comp2['smiles'],
                        'Compound2_pIC50': comp2['pIC50'],
                        'Compound2_Rgroup': comp2['rgroup_smiles'],
                        'Transform': transform,
                        'Delta_pIC50': delta,
                        'Method': method
                    })
        
        if not all_pairs:
            st.warning("No pairs generated. Check your grouping criteria.")
            return None, None
        
        pairs_df = pd.DataFrame(all_pairs)
        
        # Step 4: Analyze transformations
        status_text.text("Step 4/4: Analyzing transformations...")
        progress_bar.progress(95)
        
        # Group by transform and calculate statistics
        transform_data = []
        for transform, group in pairs_df.groupby('Transform'):
            count = len(group)
            if count >= min_transform_occurrence:
                deltas = group['Delta_pIC50'].values
                transform_data.append({
                    'Transform': transform,
                    'Count': count,
                    'Mean_ΔpIC50': np.mean(deltas),
                    'Median_ΔpIC50': np.median(deltas),
                    'Std_ΔpIC50': np.std(deltas),
                    'Min_ΔpIC50': np.min(deltas),
                    'Max_ΔpIC50': np.max(deltas),
                    'Deltas': deltas,
                    'Example_Names': f"{group.iloc[0]['Compound1_Name']}→{group.iloc[0]['Compound2_Name']}",
                    'Example_SMILES_1': group.iloc[0]['Compound1_SMILES'],
                    'Example_SMILES_2': group.iloc[0]['Compound2_SMILES'],
                    'Example_Rgroup_1': group.iloc[0]['Compound1_Rgroup'],
                    'Example_Rgroup_2': group.iloc[0]['Compound2_Rgroup'],
                    'Common_Core': group.iloc[0]['Core_SMILES']
                })
        
        if transform_data:
            transforms_df = pd.DataFrame(transform_data)
            transforms_df = transforms_df.sort_values('Mean_ΔpIC50', ascending=False)
        else:
            transforms_df = None
            st.info(f"No transforms found with {min_transform_occurrence}+ occurrences")
        
        progress_bar.progress(100)
        status_text.text("Analysis complete!")
        
        return pairs_df, transforms_df
    
    def visualize_results(pairs_df, transforms_df):
        """Create comprehensive visualizations of results"""
        
        if pairs_df is None:
            return
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Delta Distribution", "🔗 Pair Statistics", "🧪 Top Transforms", "🔍 Core Analysis"])
        
        with tab1:
            # Delta distribution
            fig, ax = plt.subplots(figsize=(10, 4))
            
            # Histogram
            ax.hist(pairs_df['Delta_pIC50'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            
            # Add statistics lines
            mean_delta = pairs_df['Delta_pIC50'].mean()
            median_delta = pairs_df['Delta_pIC50'].median()
            
            ax.axvline(mean_delta, color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {mean_delta:.2f}')
            ax.axvline(median_delta, color='green', linestyle='--', linewidth=2,
                      label=f'Median: {median_delta:.2f}')
            ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
            
            ax.set_xlabel('ΔpIC50', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title('Distribution of ΔpIC50 Values', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add statistics box
            stats_text = f"""Total Pairs: {len(pairs_df)}
Mean Δ: {mean_delta:.2f}
Std Δ: {pairs_df['Delta_pIC50'].std():.2f}
Range: [{pairs_df['Delta_pIC50'].min():.2f}, {pairs_df['Delta_pIC50'].max():.2f}]
Positive Δ: {(pairs_df['Delta_pIC50'] > 0).sum()} ({100*(pairs_df['Delta_pIC50'] > 0).mean():.1f}%)
Significant (|Δ|>1): {(abs(pairs_df['Delta_pIC50']) > 1).sum()}"""
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with tab2:
            # Pair statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Pairs", len(pairs_df))
            with col2:
                st.metric("Unique Cores", pairs_df['Core_SMILES'].nunique())
            with col3:
                positive_pairs = (pairs_df['Delta_pIC50'] > 0).sum()
                st.metric("Positive Δ", f"{positive_pairs} ({100*positive_pairs/len(pairs_df):.1f}%)")
            with col4:
                sig_pairs = (abs(pairs_df['Delta_pIC50']) > 1).sum()
                st.metric("|Δ| > 1", f"{sig_pairs} ({100*sig_pairs/len(pairs_df):.1f}%)")
            
            # Show pair table
            st.subheader("Sample Pairs (First 20)")
            display_cols = ['Compound1_Name', 'Compound1_pIC50', 'Compound2_Name', 
                          'Compound2_pIC50', 'Delta_pIC50', 'Transform']
            display_df = pairs_df[display_cols].head(20).copy()
            display_df['Transform'] = display_df['Transform'].str[:50] + '...'  # Truncate
            st.dataframe(display_df)
        
        with tab3:
            if transforms_df is not None and len(transforms_df) > 0:
                # Show top transforms
                top_n = min(n_top_transforms, len(transforms_df))
                
                for idx, (_, row) in enumerate(transforms_df.head(top_n).iterrows()):
                    with st.container():
                        st.markdown(f"### Transform #{idx+1}")
                        
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            # Transform display
                            st.markdown("**Chemical Transform:**")
                            st.code(row['Transform'], language='text')
                            
                            # Quick stats
                            st.metric("Occurrences", row['Count'])
                            st.metric("Mean ΔpIC50", f"{row['Mean_ΔpIC50']:.2f} ± {row['Std_ΔpIC50']:.2f}")
                            st.metric("Range", f"[{row['Min_ΔpIC50']:.2f}, {row['Max_ΔpIC50']:.2f}]")
                        
                        with col2:
                            # Detailed statistics
                            fig, ax = plt.subplots(figsize=(8, 3))
                            
                            # Box plot with individual points
                            ax.boxplot(row['Deltas'], vert=False, widths=0.6)
                            
                            # Add individual points
                            y = np.ones(len(row['Deltas'])) + np.random.normal(0, 0.02, len(row['Deltas']))
                            ax.scatter(row['Deltas'], y, alpha=0.6, s=30, color='red')
                            
                            ax.axvline(row['Mean_ΔpIC50'], color='blue', linestyle='--', label=f'Mean: {row["Mean_ΔpIC50"]:.2f}')
                            ax.axvline(0, color='black', linestyle='-', alpha=0.3)
                            
                            ax.set_xlabel('ΔpIC50')
                            ax.set_yticks([])
                            ax.set_title(f"Distribution (n={row['Count']})")
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Example pair
                            st.markdown(f"**Example Pair:** {row['Example_Names']}")
                        
                        # Visualize the transform if RDKit is available
                        if show_fragment_images:
                            try:
                                # Create molecules for visualization
                                rgroup1 = Chem.MolFromSmiles(row['Example_Rgroup_1'].replace('[*]', '[#0]'))
                                rgroup2 = Chem.MolFromSmiles(row['Example_Rgroup_2'].replace('[*]', '[#0]'))
                                core = Chem.MolFromSmiles(row['Common_Core'].replace('[*]', '[#0]'))
                                
                                if rgroup1 and rgroup2 and core:
                                    # Highlight attachment points
                                    for mol in [rgroup1, rgroup2, core]:
                                        for atom in mol.GetAtoms():
                                            if atom.GetAtomicNum() == 0:
                                                atom.SetProp("atomNote", "*")
                                    
                                    img = Draw.MolsToGridImage(
                                        [rgroup1, core, rgroup2],
                                        molsPerRow=3,
                                        subImgSize=(250, 200),
                                        legends=["R-group 1", "Common Core", "R-group 2"]
                                    )
                                    st.image(img, caption="Transform Visualization", use_container_width=True)
                            except:
                                pass
                        
                        st.markdown("---")
            else:
                st.info("No frequent transforms found. Try reducing the minimum occurrence threshold.")
        
        with tab4:
            # Core analysis
            st.subheader("Core Size Analysis")
            
            if 'Core_Atoms' in pairs_df.columns:
                fig, ax = plt.subplots(figsize=(10, 4))
                
                # Histogram of core sizes
                ax.hist(pairs_df['Core_Atoms'], bins=20, alpha=0.7, color='purple', edgecolor='black')
                ax.set_xlabel('Core Atoms')
                ax.set_ylabel('Number of Pairs')
                ax.set_title('Distribution of Core Sizes')
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Top cores by frequency
                st.subheader("Top 10 Most Frequent Cores")
                core_stats = pairs_df['Core_SMILES'].value_counts().head(10)
                
                for idx, (core, count) in enumerate(core_stats.items()):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.text(f"{idx+1}. {core[:80]}...")
                    with col2:
                        st.metric("Pairs", count)
            
            # Correlation analysis
            st.subheader("Core Size vs ΔpIC50")
            if 'Core_Atoms' in pairs_df.columns:
                fig, ax = plt.subplots(figsize=(8, 4))
                
                ax.scatter(pairs_df['Core_Atoms'], pairs_df['Delta_pIC50'], 
                          alpha=0.5, s=30)
                ax.set_xlabel('Core Atoms')
                ax.set_ylabel('ΔpIC50')
                ax.set_title('Core Size vs Activity Change')
                ax.grid(True, alpha=0.3)
                
                # Add trend line
                z = np.polyfit(pairs_df['Core_Atoms'], pairs_df['Delta_pIC50'], 1)
                p = np.poly1d(z)
                x_range = np.linspace(pairs_df['Core_Atoms'].min(), pairs_df['Core_Atoms'].max(), 100)
                ax.plot(x_range, p(x_range), "r--", alpha=0.8)
                
                plt.tight_layout()
                st.pyplot(fig)
    
    def create_example_dataset():
        """Create an example dataset for testing single-cut MMPA"""
        # Example set with clear single-cut opportunities
        example_smiles = [
            # Benzene derivatives - clear single cuts
            "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin-like
            "CC(=O)Oc1ccccc1C(=O)N",  # Amide derivative
            "CC(=O)Oc1ccccc1C(=O)OC",  # Methyl ester
            "CNC(=O)Oc1ccccc1C(=O)O",  # N-methyl variant
            
            # Phenyl ring with various substituents
            "Cc1ccccc1C(=O)O",  # Toluic acid
            "Cc1ccccc1C(=O)N",  # Toluamide
            "Cc1ccccc1C(=O)OC",  # Methyl toluate
            "OCc1ccccc1C(=O)O",  # Hydroxy derivative
            
            # Biphenyl system
            "O=C(O)c1ccccc1-c2ccccc2",  # Biphenyl-4-carboxylic acid
            "O=C(N)c1ccccc1-c2ccccc2",  # Biphenyl-4-carboxamide
            "O=C(OC)c1ccccc1-c2ccccc2",  # Methyl biphenyl-4-carboxylate
            
            # Simple aliphatic chains
            "CCCCCC(=O)O",  # Heptanoic acid
            "CCCCCC(=O)N",  # Heptanamide
            "CCCCCC(=O)OC",  # Methyl heptanoate
            
            # Heterocyclic examples
            "O=c1cccnc1O",  # Hydroxypyridine
            "O=c1cccnc1N",  # Aminopyridine
            "O=c1cccnc1OC",  # Methoxypyridine
        ]
        
        names = [f"Test_{i+1}" for i in range(len(example_smiles))]
        # Create realistic pIC50 values with SAR
        np.random.seed(42)
        base_potency = 5.0
        
        # Add systematic effects for different groups
        pIC50_values = []
        for smi in example_smiles:
            potency = base_potency
            
            # Acid groups generally better
            if 'C(=O)O' in smi and not 'C(=O)OC' in smi:
                potency += 0.5
            
            # Amides moderately good
            if 'C(=O)N' in smi:
                potency += 0.2
            
            # Esters generally worse
            if 'C(=O)OC' in smi:
                potency -= 0.3
            
            # Add some noise
            potency += np.random.normal(0, 0.2)
            
            pIC50_values.append(max(4.0, min(8.0, potency)))
        
        example_df = pd.DataFrame({
            'SMILES': example_smiles,
            'Name': names,
            'pIC50': pIC50_values
        })
        
        return example_df

# Main application
if not RDKIT_AVAILABLE:
    st.error("RDKit is not available. Please install it to use this tool.")
    st.info("Install with: `pip install rdkit-pypi numpy<2`")
    
else:
    # Main content area
    if uploaded_file is None:
        # Welcome screen with option to use example data
        st.markdown("""
        ## Welcome to the Advanced MMP Analysis Tool (Single Cut)
        
        This tool performs **Matched Molecular Pair (MMP)** analysis using **single cuts only** via RDKit's MMPA.
        
        ### 🎯 **Available Methods:**
        
        1. **Standard Single Cut** - Single non-ring bond cuts only
        2. **Side-chain Only** - Only terminal bond cuts  
        3. **Exhaustive Single Cut** - All single bond cuts including rings
        
        ### 🪓 **What are single cuts?**
        - Cuts one bond to separate molecule into core + R-group
        - Most chemically interpretable transforms
        - Avoids complex fragmentation patterns
        - Ideal for medicinal chemistry optimization
        
        ### 📊 **What you need:**
        - CSV file with SMILES strings
        - Optional: pIC50 values, compound names
        - Minimum 10 compounds for meaningful analysis
        
        ### 🚀 **Quick Start:**
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📁 Use Example Dataset", type="primary"):
                example_df = create_example_dataset()
                
                # Save to session state
                st.session_state.example_data = example_df
                st.session_state.use_example = True
                st.rerun()
        
        with col2:
            st.markdown("⬅️ **Or upload your own CSV file in the sidebar**")
        
        # Show example data format
        with st.expander("📋 Example Data Format"):
            example_df = create_example_dataset()
            st.dataframe(example_df)
            
            st.download_button(
                label="📥 Download Example CSV",
                data=example_df.to_csv(index=False),
                file_name="example_mmp_single_cut.csv",
                mime="text/csv"
            )
    
    else:
        # Load and process data
        with st.spinner("Loading and preprocessing data..."):
            df = load_and_preprocess_data(uploaded_file)
        
        if df is not None and len(df) > 0:
            # Show dataset overview
            st.markdown('<h2 class="section-header">📊 Dataset Overview</h2>', unsafe_allow_html=True)
            
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Compounds", len(df))
            col2.metric("Avg pIC50", f"{df['pIC50'].mean():.2f}")
            col3.metric("Avg MW", f"{df['MW'].mean():.1f}")
            col4.metric("Avg LogP", f"{df['LogP'].mean():.2f}")
            col5.metric("Avg Rotatable", f"{df['RotatableBonds'].mean():.1f}")
            
            # Show property distributions
            with st.expander("View Molecular Properties"):
                fig, axes = plt.subplots(2, 3, figsize=(12, 6))
                
                properties = ['pIC50', 'MW', 'LogP', 'HBA', 'HBD', 'RotatableBonds']
                titles = ['pIC50', 'Molecular Weight', 'LogP', 'H-Bond Acceptors', 
                         'H-Bond Donors', 'Rotatable Bonds']
                
                for idx, (prop, title, ax) in enumerate(zip(properties, titles, axes.flat)):
                    ax.hist(df[prop].dropna(), bins=20, alpha=0.7, color='steelblue', edgecolor='black')
                    ax.set_xlabel(title)
                    ax.set_ylabel('Frequency')
                    ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Show data table
                st.dataframe(df[['Name', 'SMILES', 'pIC50', 'MW', 'LogP', 'RotatableBonds']].head(10))
            
            # Perform MMP analysis
            st.markdown('<h2 class="section-header">🔍 Single-Cut MMP Analysis</h2>', unsafe_allow_html=True)
            
            # Method description
            method_desc = {
                "Standard Single Cut": "Single non-ring bond cuts only. Most chemically reasonable.",
                "Side-chain Only": "Only cuts terminal bonds. Very conservative.",
                "Exhaustive Single Cut": "All single bond cuts including rings. Most comprehensive."
            }
            
            st.info(f"**Selected Method:** {mmpa_method}")
            st.caption(method_desc[mmpa_method])
            
            # Analysis controls
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if st.button("🚀 Run Single-Cut MMP Analysis", type="primary"):
                    with st.spinner("Running single-cut MMP analysis..."):
                        pairs_df, transforms_df = perform_mmp_analysis_single_cut(
                            df, 
                            mmpa_method,
                            min_pairs_per_core,
                            show_detailed_debug
                        )
                        
                        # Store results in session state
                        st.session_state.pairs_df = pairs_df
                        st.session_state.transforms_df = transforms_df
                        st.session_state.analysis_method = mmpa_method
                        
                        # Show results if available
                        if pairs_df is not None:
                            st.success(f"✅ Generated {len(pairs_df)} molecular pairs using {mmpa_method}!")
                            
                            # Display results
                            visualize_results(pairs_df, transforms_df)
                            
                            # Export options
                            if export_all_data:
                                st.markdown('<h2 class="section-header">💾 Export Results</h2>', unsafe_allow_html=True)
                                
                                col_exp1, col_exp2, col_exp3 = st.columns(3)
                                
                                with col_exp1:
                                    if pairs_df is not None:
                                        csv_pairs = pairs_df.to_csv(index=False)
                                        st.download_button(
                                            label="📥 Download Pairs (CSV)",
                                            data=csv_pairs,
                                            file_name=f"mmp_singlecut_pairs_{mmpa_method}.csv",
                                            mime="text/csv"
                                        )
                                
                                with col_exp2:
                                    if transforms_df is not None:
                                        csv_transforms = transforms_df.to_csv(index=False)
                                        st.download_button(
                                            label="📥 Download Transforms (CSV)",
                                            data=csv_transforms,
                                            file_name=f"mmp_singlecut_transforms_{mmpa_method}.csv",
                                            mime="text/csv"
                                        )
                                
                                with col_exp3:
                                    # Combined Excel file
                                    output = io.BytesIO()
                                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                        df.to_excel(writer, sheet_name='Input_Data', index=False)
                                        if pairs_df is not None:
                                            pairs_df.to_excel(writer, sheet_name='SingleCut_Pairs', index=False)
                                        if transforms_df is not None:
                                            transforms_df.to_excel(writer, sheet_name='SingleCut_Transforms', index=False)
                                    
                                    st.download_button(
                                        label="📥 Download Full Analysis (Excel)",
                                        data=output.getvalue(),
                                        file_name=f"mmp_singlecut_analysis_{mmpa_method}.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                    )
                        else:
                            st.error("❌ No molecular pairs were generated. Try the following:")
                            st.markdown(f"""
                            ### Troubleshooting Tips:
                            
                            1. **Reduce minimum requirements**:
                               - Minimum compounds per core (currently {min_pairs_per_core})
                               - Minimum transform occurrences (currently {min_transform_occurrence})
                            
                            2. **Adjust fragment filters**:
                               - Reduce minimum core atoms (currently {min_core_atoms})
                               - Increase maximum R-group atoms (currently {max_rgroup_atoms})
                            
                            3. **Try a different method**:
                               - **Exhaustive Single Cut** is most likely to generate pairs
                               - **Side-chain Only** works if you have terminal modifications
                            
                            4. **Check your data**:
                               - Ensure compounds share common scaffolds
                               - Verify SMILES are valid
                               - More compounds (>20) increase chances
                            
                            5. **Use the example dataset** to verify the tool works
                            """)
            
            with col2:
                if st.button("🔄 Reset Analysis"):
                    if 'pairs_df' in st.session_state:
                        del st.session_state.pairs_df
                    if 'transforms_df' in st.session_state:
                        del st.session_state.transforms_df
                    if 'analysis_method' in st.session_state:
                        del st.session_state.analysis_method
                    st.rerun()
            
            # Show previously generated results if available
            if 'pairs_df' in st.session_state:
                st.markdown(f"### 📋 Previously Generated Results ({st.session_state.get('analysis_method', 'Unknown Method')})")
                visualize_results(st.session_state.pairs_df, st.session_state.transforms_df)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; font-size: 0.9rem;">
    <p>Advanced MMP Analysis Tool v3.0 (Single Cut) | Built with RDKit MMPA and Streamlit</p>
    <p>For research use only | <a href="https://www.rdkit.org/docs/source/rdkit.Chem.rdMMPA.html" target="_blank">RDKit MMPA Documentation</a></p>
</div>
""", unsafe_allow_html=True)
