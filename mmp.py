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
    page_title="Advanced MMP Analysis Tool",
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
st.markdown('<h1 class="main-header">🧪 Advanced MMP Analysis Tool</h1>', unsafe_allow_html=True)

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
        
        # Pair generation strategy
        st.markdown("#### 🔗 Pair Generation Strategy")
        pair_strategy = st.selectbox(
            "Select pair generation method",
            ["Core-based (Standard)", "Fragment-based (Comprehensive)", "Murcko Scaffold", "All Pairs (Debug)"],
            help="Core-based: Groups by exact core SMILES\nFragment-based: Uses all possible fragments\nMurcko: Groups by Bemis-Murcko scaffolds"
        )
        
        # Minimum requirements
        st.markdown("#### 📊 Minimum Requirements")
        min_pairs_per_core = st.slider("Minimum compounds per core", 2, 10, 3,
                                      help="Minimum number of compounds sharing the same core")
        min_transform_occurrence = st.slider("Minimum transform occurrences", 1, 20, 2,
                                           help="Minimum occurrences for statistical significance")
        
        # Fragmentation settings
        st.markdown("#### 🪓 Fragmentation Settings")
        max_bond_cuts = st.slider("Maximum bond cuts", 1, 3, 2,
                                 help="Maximum number of bonds to cut simultaneously")
        allow_ring_cuts = st.checkbox("Allow ring cuts", value=False,
                                     help="Allow cutting bonds in rings (may produce small fragments)")
        
        # Property filters
        st.markdown("#### 🔍 Property Filters")
        min_mw = st.number_input("Minimum MW", 0.0, 1000.0, 100.0, 10.0,
                                help="Minimum molecular weight")
        max_mw = st.number_input("Maximum MW", 100.0, 2000.0, 500.0, 10.0,
                                help="Maximum molecular weight")
        
        # Display options
        st.markdown("### 👀 Display Options")
        n_top_transforms = st.slider("Top transforms to display", 1, 50, 10)
        show_detailed_debug = st.checkbox("Show detailed debug info", value=False)
        show_fragment_analysis = st.checkbox("Show fragment analysis", value=True)
        
        # Export options
        st.markdown("### 💾 Export")
        export_all_data = st.checkbox("Export all data", value=True)
        
        # About
        with st.expander("ℹ️ About This Tool"):
            st.markdown("""
            **Advanced MMP Analysis Tool**
            
            This tool uses multiple strategies for MMP analysis:
            
            1. **Core-based**: Standard MMP with exact core matching
            2. **Fragment-based**: Comprehensive analysis using all fragments
            3. **Murcko Scaffold**: Groups by Bemis-Murcko scaffolds
            4. **All Pairs**: Debug mode to see all possible pairs
            
            **Tips for success:**
            - Start with larger datasets (>50 compounds)
            - Use lower minimum requirements initially
            - Try different fragmentation settings
            - Check debug info if no pairs are found
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
            
            st.success(f"Loaded {len(final_df)} valid compounds")
            return final_df
            
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None
    
    def get_murcko_scaffold(mol):
        """Get Murcko scaffold SMILES for a molecule"""
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold)
        except:
            return None
    
    def generate_fragments_advanced(mol, max_cuts=2, allow_ring_cuts=False):
        """Generate fragments using RDKit MMPA with advanced options"""
        fragments = []
        
        try:
            # Use RDKit's FragmentMol for systematic fragmentation
            results = rdMMPA.FragmentMol(
                mol,
                maxCuts=max_cuts,
                maxCutBonds=20,
                pattern="[*:1]~[*:2]",
                resultsAsMols=False
            )
            
            for core_smiles, rgroup_smiles in results:
                # Clean the SMILES for consistency
                core_smiles = core_smiles.replace('[*:1]', '*').replace('[*:2]', '*')
                rgroup_smiles = rgroup_smiles.replace('[*:1]', '*').replace('[*:2]', '*')
                
                # Skip very small cores
                core_mol = Chem.MolFromSmiles(core_smiles)
                if core_mol and core_mol.GetNumAtoms() >= 5:
                    fragments.append({
                        'core_smiles': core_smiles,
                        'rgroup_smiles': rgroup_smiles,
                        'core_num_atoms': core_mol.GetNumAtoms()
                    })
                    
        except Exception as e:
            # Fallback to single bond cutting
            try:
                for bond in mol.GetBonds():
                    if bond.GetBondType() == Chem.BondType.SINGLE:
                        if not allow_ring_cuts and bond.IsInRing():
                            continue
                        
                        emol = Chem.EditableMol(mol)
                        emol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                        frag_mol = emol.GetMol()
                        
                        frags = Chem.GetMolFrags(frag_mol, asMols=True, sanitizeFrags=False)
                        if len(frags) == 2:
                            # Sort by size (largest first as core)
                            frags_sorted = sorted(frags, key=lambda x: x.GetNumAtoms(), reverse=True)
                            core_smiles = Chem.MolToSmiles(frags_sorted[0])
                            rgroup_smiles = Chem.MolToSmiles(frags_sorted[1])
                            
                            # Add attachment point
                            core_smiles = core_smiles + '[*]'
                            rgroup_smiles = rgroup_smiles + '[*]'
                            
                            fragments.append({
                                'core_smiles': core_smiles,
                                'rgroup_smiles': rgroup_smiles,
                                'core_num_atoms': frags_sorted[0].GetNumAtoms()
                            })
            except:
                pass
        
        return fragments
    
    def perform_mmp_analysis_robust(df, strategy, min_pairs_per_core, show_debug=False):
        """Perform MMP analysis with robust pair generation"""
        
        st.info(f"Starting MMP analysis using {strategy} strategy...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Generate fragments or scaffolds
        status_text.text("Step 1/4: Generating molecular representations...")
        progress_bar.progress(25)
        
        compound_data = []
        
        for idx, row in df.iterrows():
            mol = row['mol']
            
            if strategy == "Murcko Scaffold":
                # Use Murcko scaffolds
                scaffold_smiles = get_murcko_scaffold(mol)
                if scaffold_smiles:
                    compound_data.append({
                        'idx': idx,
                        'name': row['Name'],
                        'smiles': row['SMILES'],
                        'pIC50': row['pIC50'],
                        'core': scaffold_smiles,
                        'rgroup': '',  # Not applicable for scaffolds
                        'type': 'scaffold'
                    })
            
            elif strategy == "All Pairs (Debug)":
                # Just store molecule info
                compound_data.append({
                    'idx': idx,
                    'name': row['Name'],
                    'smiles': row['SMILES'],
                    'pIC50': row['pIC50'],
                    'core': f"ALL_{idx}",
                    'rgroup': '',
                    'type': 'all_pairs'
                })
            
            else:
                # Generate fragments
                fragments = generate_fragments_advanced(
                    mol, 
                    max_cuts=max_bond_cuts,
                    allow_ring_cuts=allow_ring_cuts
                )
                
                for frag in fragments:
                    compound_data.append({
                        'idx': idx,
                        'name': row['Name'],
                        'smiles': row['SMILES'],
                        'pIC50': row['pIC50'],
                        'core': frag['core_smiles'],
                        'rgroup': frag['rgroup_smiles'],
                        'type': 'fragment',
                        'core_size': frag['core_num_atoms']
                    })
        
        if not compound_data:
            st.error("No compound data generated. Try different fragmentation settings.")
            return None, None
        
        # Debug: Show fragment statistics
        if show_debug:
            with st.expander("📊 Fragment Statistics", expanded=False):
                frag_df = pd.DataFrame(compound_data)
                st.write(f"Total representations: {len(frag_df)}")
                st.write(f"Unique cores: {frag_df['core'].nunique()}")
                st.write(f"Average fragments per compound: {len(frag_df)/len(df):.1f}")
                
                if 'core_size' in frag_df.columns:
                    st.write(f"Average core size: {frag_df['core_size'].mean():.1f} atoms")
                    st.write(f"Core size distribution:")
                    fig, ax = plt.subplots(figsize=(8, 3))
                    ax.hist(frag_df['core_size'].dropna(), bins=20, alpha=0.7, color='blue')
                    ax.set_xlabel('Core atoms')
                    ax.set_ylabel('Frequency')
                    st.pyplot(fig)
        
        # Step 2: Group compounds by core
        status_text.text("Step 2/4: Grouping compounds by common features...")
        progress_bar.progress(50)
        
        core_groups = defaultdict(list)
        for data in compound_data:
            core_groups[data['core']].append(data)
        
        # Filter groups by minimum size
        valid_groups = {core: comps for core, comps in core_groups.items() 
                       if len(comps) >= min_pairs_per_core}
        
        if not valid_groups:
            st.warning(f"No cores found with {min_pairs_per_core}+ compounds. Try reducing threshold.")
            return None, None
        
        # Debug: Show group statistics
        if show_debug:
            with st.expander("📊 Group Statistics", expanded=False):
                st.write(f"Total groups: {len(core_groups)}")
                st.write(f"Valid groups (≥{min_pairs_per_core} compounds): {len(valid_groups)}")
                
                group_sizes = [len(comps) for comps in valid_groups.values()]
                if group_sizes:
                    st.write(f"Average group size: {np.mean(group_sizes):.1f}")
                    st.write(f"Largest group: {max(group_sizes)} compounds")
                    
                    # Show top 10 largest groups
                    st.subheader("Top 10 largest groups:")
                    top_groups = sorted(valid_groups.items(), key=lambda x: len(x[1]), reverse=True)[:10]
                    for core, comps in top_groups:
                        st.write(f"Core: {core[:50]}... | Compounds: {len(comps)}")
        
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
                    if comp1['idx'] == comp2['idx']:
                        continue
                    
                    # Calculate delta
                    delta = comp2['pIC50'] - comp1['pIC50']
                    
                    # Create transform string
                    if strategy in ["Core-based (Standard)", "Fragment-based (Comprehensive)"]:
                        transform = f"{comp1['rgroup']}>>{comp2['rgroup']}"
                    else:
                        transform = f"{core}_transform"
                    
                    # Store pair
                    all_pairs.append({
                        'Core': core,
                        'SMILES_1': comp1['smiles'],
                        'Name_1': comp1['name'],
                        'pIC50_1': comp1['pIC50'],
                        'SMILES_2': comp2['smiles'],
                        'Name_2': comp2['name'],
                        'pIC50_2': comp2['pIC50'],
                        'Transform': transform,
                        'Delta': delta,
                        'Strategy': strategy
                    })
        
        if not all_pairs:
            st.warning("No pairs generated. Check your grouping criteria.")
            return None, None
        
        pairs_df = pd.DataFrame(all_pairs)
        
        # Step 4: Analyze transformations
        status_text.text("Step 4/4: Analyzing transformations...")
        progress_bar.progress(95)
        
        if strategy == "All Pairs (Debug)":
            # For debug mode, just return pairs
            transforms_df = None
        else:
            # Group by transform and calculate statistics
            transform_data = []
            for transform, group in pairs_df.groupby('Transform'):
                count = len(group)
                if count >= min_transform_occurrence:
                    deltas = group['Delta'].values
                    transform_data.append({
                        'Transform': transform,
                        'Count': count,
                        'Mean_ΔpIC50': np.mean(deltas),
                        'Median_ΔpIC50': np.median(deltas),
                        'Std_ΔpIC50': np.std(deltas),
                        'Min_ΔpIC50': np.min(deltas),
                        'Max_ΔpIC50': np.max(deltas),
                        'Deltas': deltas,
                        'Example_Names': f"{group.iloc[0]['Name_1']}→{group.iloc[0]['Name_2']}",
                        'Example_SMILES_1': group.iloc[0]['SMILES_1'],
                        'Example_SMILES_2': group.iloc[0]['SMILES_2']
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
        tab1, tab2, tab3 = st.tabs(["📈 Delta Distribution", "🔗 Pair Statistics", "🧪 Top Transforms"])
        
        with tab1:
            # Delta distribution
            fig, ax = plt.subplots(figsize=(10, 4))
            
            # Histogram
            ax.hist(pairs_df['Delta'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            
            # Add statistics lines
            mean_delta = pairs_df['Delta'].mean()
            median_delta = pairs_df['Delta'].median()
            
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
Std Δ: {pairs_df['Delta'].std():.2f}
Range: [{pairs_df['Delta'].min():.2f}, {pairs_df['Delta'].max():.2f}]
Positive Δ: {(pairs_df['Delta'] > 0).sum()} ({100*(pairs_df['Delta'] > 0).mean():.1f}%)
Significant (|Δ|>1): {(abs(pairs_df['Delta']) > 1).sum()}"""
            
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
                st.metric("Unique Cores", pairs_df['Core'].nunique())
            with col3:
                positive_pairs = (pairs_df['Delta'] > 0).sum()
                st.metric("Positive Δ", f"{positive_pairs} ({100*positive_pairs/len(pairs_df):.1f}%)")
            with col4:
                sig_pairs = (abs(pairs_df['Delta']) > 1).sum()
                st.metric("|Δ| > 1", f"{sig_pairs} ({100*sig_pairs/len(pairs_df):.1f}%)")
            
            # Show pair table
            st.subheader("Sample Pairs (First 20)")
            display_cols = ['Name_1', 'pIC50_1', 'Name_2', 'pIC50_2', 'Delta', 'Core']
            display_df = pairs_df[display_cols].head(20).copy()
            display_df['Core'] = display_df['Core'].str[:50] + '...'  # Truncate long cores
            st.dataframe(display_df)
            
            # Core size distribution
            if 'Core' in pairs_df.columns:
                st.subheader("Core Analysis")
                core_stats = pairs_df['Core'].value_counts().head(10)
                fig, ax = plt.subplots(figsize=(10, 4))
                core_stats.plot(kind='bar', ax=ax, color='steelblue')
                ax.set_xlabel('Core (truncated)')
                ax.set_ylabel('Number of Pairs')
                ax.set_title('Top 10 Most Frequent Cores')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
        
        with tab3:
            if transforms_df is not None and len(transforms_df) > 0:
                # Show top transforms
                top_n = min(n_top_transforms, len(transforms_df))
                
                for idx, (_, row) in enumerate(transforms_df.head(top_n).iterrows()):
                    with st.container():
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            # Simple transform display
                            st.markdown(f"**Transform {idx+1}**")
                            st.code(row['Transform'], language='text')
                            
                            # Quick stats
                            st.metric("Occurrences", row['Count'])
                            st.metric("Mean Δ", f"{row['Mean_ΔpIC50']:.2f}")
                        
                        with col2:
                            # Detailed statistics
                            fig, ax = plt.subplots(figsize=(8, 2))
                            
                            # Strip plot
                            y = np.random.normal(0, 0.1, len(row['Deltas']))
                            ax.scatter(row['Deltas'], y, alpha=0.6, s=50)
                            ax.axvline(row['Mean_ΔpIC50'], color='red', linestyle='--', label='Mean')
                            ax.axvline(0, color='black', linestyle='-', alpha=0.3)
                            
                            ax.set_xlabel('ΔpIC50')
                            ax.set_yticks([])
                            ax.set_title(f"Distribution (n={row['Count']})")
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Example pair
                            st.markdown(f"**Example:** {row['Example_Names']}")
                        
                        st.markdown("---")
            else:
                st.info("No frequent transforms found. Try reducing the minimum occurrence threshold.")
    
    def create_example_dataset():
        """Create an example dataset for testing"""
        example_smiles = [
            "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
            "CC(=O)OC1=CC=CC=C1C(=O)N",  # Aspirin-like with NH2
            "CC(=O)OC1=CC=C(C=C1)C(=O)O",  # Ortho-substituted
            "CC(=O)OC1=CC=CC=C1C(=O)OC",  # Methyl ester
            "CC(=O)OC1=CC=CC=C1C(=O)NC",  # N-methyl amide
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)CC",  # Caffeine with ethyl
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)CCO",  # Caffeine with OH
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)OC",  # Ibuprofen methyl ester
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)N",  # Ibuprofen amide
            "C1=CC=C(C=C1)C=O",  # Benzaldehyde
            "C1=CC=C(C=C1)C(=O)O",  # Benzoic acid
            "C1=CC=C(C=C1)C(=O)N",  # Benzamide
            "C1=CC=C(C=C1)C(=O)OC",  # Methyl benzoate
        ]
        
        names = [f"Test_{i+1}" for i in range(len(example_smiles))]
        # Create realistic pIC50 values with some structure-activity relationship
        pIC50_values = []
        base_values = [4.5, 5.0, 4.8, 4.3, 5.2,  # Aspirin series
                      5.2, 5.5, 5.8, 6.1, 5.9,  # Caffeine series
                      6.1, 5.8, 6.3, 3.8, 4.0]  # Other series
        
        # Add some noise
        np.random.seed(42)
        noise = np.random.normal(0, 0.3, len(example_smiles))
        pIC50_values = base_values + noise
        
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
        ## Welcome to the Advanced MMP Analysis Tool
        
        This tool performs **Matched Molecular Pair (MMP)** analysis using multiple strategies:
        
        ### 🎯 **Available Strategies:**
        
        1. **Core-based (Standard)** - Exact core matching
        2. **Fragment-based (Comprehensive)** - All possible fragments
        3. **Murcko Scaffold** - Bemis-Murcko scaffold grouping  
        4. **All Pairs (Debug)** - Debug mode for troubleshooting
        
        ### 📊 **What you need:**
        - CSV file with SMILES strings
        - Optional: pIC50 values, compound names
        - Minimum 10-20 compounds for meaningful analysis
        
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
                file_name="example_mmp_data.csv",
                mime="text/csv"
            )
    
    else:
        # Load and process data
        with st.spinner("Loading and preprocessing data..."):
            df = load_and_preprocess_data(uploaded_file)
        
        if df is not None and len(df) > 0:
            # Show dataset overview
            st.markdown('<h2 class="section-header">📊 Dataset Overview</h2>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Compounds", len(df))
            col2.metric("Avg pIC50", f"{df['pIC50'].mean():.2f}")
            col3.metric("Avg MW", f"{df['MW'].mean():.1f}")
            col4.metric("Avg LogP", f"{df['LogP'].mean():.2f}")
            
            # Show property distributions
            with st.expander("View Molecular Properties"):
                fig, axes = plt.subplots(2, 3, figsize=(12, 6))
                
                properties = ['pIC50', 'MW', 'LogP', 'HBA', 'HBD', 'TPSA']
                titles = ['pIC50', 'Molecular Weight', 'LogP', 'H-Bond Acceptors', 
                         'H-Bond Donors', 'TPSA']
                
                for idx, (prop, title, ax) in enumerate(zip(properties, titles, axes.flat)):
                    ax.hist(df[prop].dropna(), bins=20, alpha=0.7, color='steelblue', edgecolor='black')
                    ax.set_xlabel(title)
                    ax.set_ylabel('Frequency')
                    ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Show data table
                st.dataframe(df[['Name', 'SMILES', 'pIC50', 'MW', 'LogP']].head(10))
            
            # Perform MMP analysis
            st.markdown('<h2 class="section-header">🔍 MMP Analysis</h2>', unsafe_allow_html=True)
            
            # Analysis controls
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if st.button("🚀 Run MMP Analysis", type="primary"):
                    with st.spinner("Running MMP analysis..."):
                        pairs_df, transforms_df = perform_mmp_analysis_robust(
                            df, 
                            pair_strategy,
                            min_pairs_per_core,
                            show_detailed_debug
                        )
                        
                        # Store results in session state
                        st.session_state.pairs_df = pairs_df
                        st.session_state.transforms_df = transforms_df
                        
                        # Show results if available
                        if pairs_df is not None:
                            st.success(f"✅ Generated {len(pairs_df)} molecular pairs!")
                            
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
                                            file_name=f"mmp_pairs_{pair_strategy}.csv",
                                            mime="text/csv"
                                        )
                                
                                with col_exp2:
                                    if transforms_df is not None:
                                        csv_transforms = transforms_df.to_csv(index=False)
                                        st.download_button(
                                            label="📥 Download Transforms (CSV)",
                                            data=csv_transforms,
                                            file_name=f"mmp_transforms_{pair_strategy}.csv",
                                            mime="text/csv"
                                        )
                                
                                with col_exp3:
                                    # Combined Excel file
                                    output = io.BytesIO()
                                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                        df.to_excel(writer, sheet_name='Input_Data', index=False)
                                        if pairs_df is not None:
                                            pairs_df.to_excel(writer, sheet_name='All_Pairs', index=False)
                                        if transforms_df is not None:
                                            transforms_df.to_excel(writer, sheet_name='Transforms', index=False)
                                    
                                    st.download_button(
                                        label="📥 Download Full Analysis (Excel)",
                                        data=output.getvalue(),
                                        file_name=f"mmp_analysis_{pair_strategy}.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                    )
                        else:
                            st.error("❌ No molecular pairs were generated. Try the following:")
                            st.markdown("""
                            ### Troubleshooting Tips:
                            
                            1. **Reduce minimum requirements** - Try lower values for:
                               - Minimum compounds per core (currently {min_pairs_per_core})
                               - Minimum transform occurrences (currently {min_transform_occurrence})
                            
                            2. **Try a different strategy**:
                               - **Fragment-based** is most comprehensive
                               - **Murcko Scaffold** works well for scaffold-hopping
                               - **All Pairs (Debug)** shows what's being generated
                            
                            3. **Check your data**:
                               - Ensure pIC50 values have sufficient variation
                               - Verify SMILES are valid (check error messages)
                               - Try with more compounds (>20 recommended)
                            
                            4. **Adjust fragmentation**:
                               - Increase maximum bond cuts
                               - Allow ring cuts (if chemically relevant)
                            
                            5. **Use the example dataset** to verify the tool works
                            """.format(min_pairs_per_core=min_pairs_per_core, 
                                      min_transform_occurrence=min_transform_occurrence))
            
            with col2:
                if st.button("🔄 Reset Analysis"):
                    if 'pairs_df' in st.session_state:
                        del st.session_state.pairs_df
                    if 'transforms_df' in st.session_state:
                        del st.session_state.transforms_df
                    st.rerun()
            
            # Show previously generated results if available
            if 'pairs_df' in st.session_state:
                st.markdown("### 📋 Previously Generated Results")
                visualize_results(st.session_state.pairs_df, st.session_state.transforms_df)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; font-size: 0.9rem;">
    <p>Advanced MMP Analysis Tool v3.0 | Built with RDKit and Streamlit</p>
    <p>For research use only | <a href="https://www.rdkit.org" target="_blank">RDKit Documentation</a></p>
</div>
""", unsafe_allow_html=True)
