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

# Suppress warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="MMP Analysis Tool (RDKit MMPA)",
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
    .info-box {
        background-color: #EFF6FF;
        border-left: 4px solid #60A5FA;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header"> Matched Molecular Pair Analysis Tool (RDKit MMPA)</h1>', unsafe_allow_html=True)

# Try to import RDKit with error handling
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw, rdFMCS
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.Chem import rdDepictor
    from rdkit.Chem import rdMMPA
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
        # MMPA Parameters
        st.markdown("### ⚙️ MMPA Parameters")
        
        # Fragmentation parameters
        st.markdown("#### 🪓 Fragmentation Settings")
        max_cuts = st.slider("Maximum cuts per bond", 1, 3, 2,
                           help="Maximum number of cuts to make (1-3)")
        max_cut_bonds = st.slider("Maximum bonds to cut", 1, 20, 12,
                                help="Maximum number of bonds to cut in total")
        
        min_occurrence = st.slider("Minimum transform occurrences", 1, 50, 5,
                                  help="Minimum number of occurrences for a transform to be considered")
        
        # Molecule cleaning options
        st.markdown("### 🧹 Molecule Cleaning")
        sanitize_molecules = st.checkbox("Sanitize molecules", value=True,
                                       help="Clean molecules (recommended)")
        
        # Display options
        st.markdown("### 👀 Display Options")
        show_all_transforms = st.checkbox("Show all transformations", value=False)
        transforms_to_display = st.slider("Number of transforms to display", 1, 100, 20,
                                         disabled=show_all_transforms)
        
        # Highlighting options
        st.markdown("### 🎨 Highlighting Options")
        highlight_color_before = st.color_picker("Before transformation color", "#FF6B6B")
        highlight_color_after = st.color_picker("After transformation color", "#4ECDC4")
        
        # Analysis options
        st.markdown("### 🔬 Analysis Options")
        show_top_positive = st.checkbox("Show top positive transforms", value=True)
        show_top_negative = st.checkbox("Show top negative transforms", value=True)
        show_compound_examples = st.checkbox("Show compound examples", value=True)
        show_mcs_analysis = st.checkbox("Show Maximum Common Substructure (MCS)", value=True)
        
        # Pair filtering
        st.markdown("### 🔗 Pair Filtering")
        min_core_size = st.slider("Minimum core atoms", 5, 50, 10,
                                 help="Minimum number of atoms in common core")
        min_delta_threshold = st.number_input("Minimum |ΔpIC50| threshold", 0.0, 10.0, 0.5, 0.1,
                                            help="Minimum absolute ΔpIC50 to consider")
        
        # Save options
        st.markdown("### 💾 Export")
        save_results = st.checkbox("Save results to Excel")
        
        # Debug option
        st.markdown("### 🐛 Debug")
        show_debug_info = st.checkbox("Show debug information", value=False)
    
    # About section
    with st.expander("About RDKit MMPA"):
        st.markdown("""
        **RDKit MMPA** provides professional-grade Matched Molecular Pair analysis:
        
        **Key Features:**
        1. **Systematic fragmentation** using RDKit's MMPA module
        2. **Efficient pair generation** with configurable cut rules
        3. **Statistical analysis** of transformations
        4. **Maximum Common Substructure** visualization
        
        **Fragmentation Protocol:**
        - Bonds are systematically cut based on chemical rules
        - Fragments are canonicalized for comparison
        - Cores must meet minimum size requirements
        
        **References:**
        - RDKit Documentation: https://www.rdkit.org/docs/Cookbook.html#mmpa
        - Hussain & Rea, J. Chem. Inf. Model., 2010
        """)

# Helper functions
if RDKIT_AVAILABLE:
    @st.cache_data
    def load_data(file, sanitize=True):
        """Load and preprocess data"""
        if file is not None:
            df = pd.read_csv(file)
            
            # Check required columns
            required_cols = ['SMILES', 'pIC50']
            if not all(col in df.columns for col in required_cols):
                st.error(f"CSV must contain columns: {required_cols}")
                return None
            
            # Add Name column if not present
            if 'Name' not in df.columns:
                df['Name'] = [f"CMPD_{i+1}" for i in range(len(df))]
            
            # Convert SMILES to molecules
            molecules = []
            errors = []
            
            for idx, row in df.iterrows():
                try:
                    mol = Chem.MolFromSmiles(str(row['SMILES']))
                    if mol is None:
                        errors.append(f"Row {idx}: Invalid SMILES '{row['SMILES']}'")
                        molecules.append(None)
                        continue
                    
                    if sanitize:
                        try:
                            Chem.SanitizeMol(mol)
                        except:
                            pass
                    
                    # Get largest fragment
                    frags = Chem.GetMolFrags(mol, asMols=True)
                    if frags:
                        mol = max(frags, key=lambda x: x.GetNumAtoms())
                    
                    molecules.append(mol)
                except Exception as e:
                    errors.append(f"Row {idx}: Error processing '{row['SMILES']}' - {str(e)}")
                    molecules.append(None)
            
            df['mol'] = molecules
            
            # Show errors if any
            if errors:
                with st.expander("⚠️ Processing Errors", expanded=True):
                    for error in errors[:10]:
                        st.warning(error)
                    if len(errors) > 10:
                        st.info(f"... and {len(errors)-10} more errors")
            
            # Remove rows with invalid molecules
            valid_df = df[df['mol'].notna()].copy()
            if len(valid_df) < len(df):
                st.warning(f"Removed {len(df) - len(valid_df)} rows with invalid molecules")
            
            return valid_df
        return None
    
    def perform_rdkit_mmpa_analysis(df, max_cuts=2, max_cut_bonds=12, min_transform_occurrence=5,
                                   min_core_size=10, min_delta_threshold=0.5, show_debug=False):
        """Perform MMP analysis using RDKit's built-in MMPA functionality"""
        
        if df is None or len(df) == 0:
            return None, None, None
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Generate all possible fragments using RDKit MMPA
        status_text.text("Step 1/4: Generating molecular fragments...")
        progress_bar.progress(25)
        
        all_fragments = []
        fragment_map = defaultdict(list)  # Maps core_smiles -> list of (mol_idx, rgroup_smiles, pIC50, name)
        
        for idx, row in df.iterrows():
            mol = row['mol']
            if mol is None:
                continue
            
            try:
                # Generate fragments using RDKit MMPA
                # This is the core fragmentation function
                fragments = rdMMPA.FragmentMol(
                    mol,
                    maxCuts=max_cuts,
                    maxCutBonds=max_cut_bonds,
                    pattern="[*:1]~[*:2]",
                    resultsAsMols=False
                )
                
                for core_smiles, rgroup_smiles in fragments:
                    # Store fragment information
                    all_fragments.append({
                        'mol_idx': idx,
                        'core_smiles': core_smiles,
                        'rgroup_smiles': rgroup_smiles,
                        'pIC50': row['pIC50'],
                        'name': row['Name'],
                        'smiles': row['SMILES']
                    })
                    
                    # Group by core for efficient pair generation
                    fragment_map[core_smiles].append({
                        'mol_idx': idx,
                        'rgroup_smiles': rgroup_smiles,
                        'pIC50': row['pIC50'],
                        'name': row['Name'],
                        'smiles': row['SMILES']
                    })
                    
            except Exception as e:
                if show_debug:
                    st.write(f"Error fragmenting molecule {idx}: {e}")
                continue
        
        if show_debug:
            with st.expander("Debug: Fragment Statistics", expanded=False):
                st.write(f"Total fragments generated: {len(all_fragments)}")
                st.write(f"Unique cores: {len(fragment_map)}")
                st.write(f"Average fragments per molecule: {len(all_fragments)/len(df):.2f}")
        
        # Step 2: Generate molecular pairs
        status_text.text("Step 2/4: Generating molecular pairs...")
        progress_bar.progress(50)
        
        pairs = []
        cores_with_pairs = 0
        
        for core_smiles, fragments in fragment_map.items():
            # Only process cores with multiple compounds (at least 2)
            if len(fragments) >= 2:
                # Generate all unique pairs
                for i in range(len(fragments)):
                    for j in range(i+1, len(fragments)):
                        frag1 = fragments[i]
                        frag2 = fragments[j]
                        
                        # Skip if same molecule
                        if frag1['mol_idx'] == frag2['mol_idx']:
                            continue
                        
                        # Calculate delta pIC50
                        delta = frag2['pIC50'] - frag1['pIC50']
                        
                        # Apply threshold filter
                        if abs(delta) < min_delta_threshold:
                            continue
                        
                        # Create transform string
                        # RDKit fragments have attachment points marked with [*:1] etc.
                        # We need to standardize them
                        rgroup1 = frag1['rgroup_smiles']
                        rgroup2 = frag2['rgroup_smiles']
                        
                        # Standardize attachment points to just *
                        rgroup1_clean = rgroup1.replace('[*:1]', '*').replace('[*:2]', '*').replace('[*:3]', '*')
                        rgroup2_clean = rgroup2.replace('[*:1]', '*').replace('[*:2]', '*').replace('[*:3]', '*')
                        
                        transform = f"{rgroup1_clean}>>{rgroup2_clean}"
                        
                        # Check core size (number of atoms in core)
                        try:
                            core_mol = Chem.MolFromSmiles(core_smiles)
                            if core_mol and core_mol.GetNumAtoms() >= min_core_size:
                                pairs.append({
                                    'SMILES_1': frag1['smiles'],
                                    'Name_1': frag1['name'],
                                    'pIC50_1': frag1['pIC50'],
                                    'R_group_1': rgroup1_clean,
                                    'SMILES_2': frag2['smiles'],
                                    'Name_2': frag2['name'],
                                    'pIC50_2': frag2['pIC50'],
                                    'R_group_2': rgroup2_clean,
                                    'Core': core_smiles,
                                    'Transform': transform,
                                    'Delta': delta
                                })
                                cores_with_pairs += 1
                        except:
                            continue
        
        if show_debug:
            with st.expander("Debug: Pair Generation", expanded=False):
                st.write(f"Total pairs generated: {len(pairs)}")
                st.write(f"Cores contributing to pairs: {cores_with_pairs}")
                if pairs:
                    st.write("Sample pairs:")
                    for i, pair in enumerate(pairs[:3]):
                        st.write(f"Pair {i+1}: {pair['Name_1']} -> {pair['Name_2']}, Δ={pair['Delta']:.2f}")
                        st.write(f"  Transform: {pair['Transform']}")
        
        if not pairs:
            st.error("No molecular pairs found meeting criteria")
            return None, None, None
        
        # Create pairs DataFrame
        pairs_df = pd.DataFrame(pairs)
        
        # Step 3: Analyze transformations
        status_text.text("Step 3/4: Analyzing transformations...")
        progress_bar.progress(75)
        
        # Group by transform
        transform_data = []
        transform_groups = pairs_df.groupby('Transform')
        
        for transform, group in transform_groups:
            count = len(group)
            if count >= min_transform_occurrence:
                deltas = group['Delta'].values
                mean_delta = np.mean(deltas)
                std_delta = np.std(deltas)
                median_delta = np.median(deltas)
                
                # Get example pair
                example_pair = group.iloc[0]
                
                transform_data.append({
                    'Transform': transform,
                    'Count': count,
                    'Mean ΔpIC50': mean_delta,
                    'Std ΔpIC50': std_delta,
                    'Median ΔpIC50': median_delta,
                    'Min ΔpIC50': min(deltas),
                    'Max ΔpIC50': max(deltas),
                    'Deltas': deltas,
                    'Example_SMILES_1': example_pair['SMILES_1'],
                    'Example_SMILES_2': example_pair['SMILES_2'],
                    'Example_Names': f"{example_pair['Name_1']} → {example_pair['Name_2']}"
                })
        
        if not transform_data:
            st.warning(f"No transforms found with {min_transform_occurrence}+ occurrences")
            return pairs_df, None, None
        
        transforms_df = pd.DataFrame(transform_data)
        
        # Sort by mean delta
        transforms_df = transforms_df.sort_values('Mean ΔpIC50', ascending=False)
        
        # Step 4: Generate reaction SMARTS for visualization
        status_text.text("Step 4/4: Generating visualizations...")
        
        rxn_smarts_list = []
        for _, row in transforms_df.iterrows():
            try:
                # Parse the transform
                parts = row['Transform'].split('>>')
                if len(parts) == 2:
                    left = parts[0]
                    right = parts[1]
                    
                    # Create reaction SMARTS
                    rxn_smarts = f"{left}>>{right}"
                    rxn_smarts_list.append(rxn_smarts)
                else:
                    rxn_smarts_list.append(None)
            except:
                rxn_smarts_list.append(None)
        
        transforms_df['Reaction_SMARTS'] = rxn_smarts_list
        
        progress_bar.progress(100)
        status_text.empty()
        progress_bar.empty()
        
        return pairs_df, transforms_df, fragment_map
    
    def create_mcs_image(mol1, mol2, height=300):
        """Create image showing Maximum Common Substructure"""
        try:
            # Find MCS
            mcs = rdFMCS.FindMCS([mol1, mol2], 
                                 timeout=30,
                                 atomCompare=rdFMCS.AtomCompare.CompareAny,
                                 bondCompare=rdFMCS.BondCompare.CompareAny,
                                 matchValences=False,
                                 ringMatchesRingOnly=False,
                                 completeRingsOnly=False)
            
            if mcs.numAtoms == 0:
                return None
            
            # Create substructure from MCS
            mcs_pattern = Chem.MolFromSmarts(mcs.smartsString)
            
            # Highlight matches
            matches1 = mol1.GetSubstructMatch(mcs_pattern)
            matches2 = mol2.GetSubstructMatch(mcs_pattern)
            
            if not matches1 or not matches2:
                return None
            
            # Create drawing
            drawer = rdMolDraw2D.MolDraw2DCairo(400, height)
            opts = drawer.drawOptions()
            
            # Set up colors
            highlight_color1 = (1.0, 0.42, 0.42)  # Light red
            highlight_color2 = (0.31, 0.80, 0.76)  # Teal
            
            # Create atom highlights
            atom_colors1 = {}
            atom_colors2 = {}
            
            for idx in matches1:
                atom_colors1[idx] = highlight_color1
            
            for idx in matches2:
                atom_colors2[idx] = highlight_color2
            
            # Draw both molecules
            drawer.DrawMolecule(mol1, highlightAtoms=matches1, highlightAtomColors=atom_colors1)
            drawer.FinishDrawing()
            
            # Convert to base64
            img1 = drawer.GetDrawingText()
            
            # Draw second molecule
            drawer = rdMolDraw2D.MolDraw2DCairo(400, height)
            drawer.DrawMolecule(mol2, highlightAtoms=matches2, highlightAtomColors=atom_colors2)
            drawer.FinishDrawing()
            
            img2 = drawer.GetDrawingText()
            
            return base64.b64encode(img1).decode(), base64.b64encode(img2).decode(), mcs.smartsString
            
        except Exception as e:
            if show_debug:
                st.write(f"MCS error: {e}")
            return None
    
    def plot_delta_distribution(deltas, title="ΔpIC50 Distribution"):
        """Create distribution plot for deltas"""
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Histogram
        n, bins, patches = ax.hist(deltas, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Add vertical lines for mean and median
        mean_val = np.mean(deltas)
        median_val = np.median(deltas)
        
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
        ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        
        # Add statistics text
        stats_text = f'n = {len(deltas)}\nMean = {mean_val:.2f}\nStd = {np.std(deltas):.2f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('ΔpIC50')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def get_reaction_image(transform, width=400, height=200):
        """Create image for chemical transformation"""
        try:
            parts = transform.split('>>')
            if len(parts) != 2:
                return None
            
            # Create molecules from SMILES
            left_smiles = parts[0]
            right_smiles = parts[1]
            
            # Replace * with [*] for proper parsing
            left_smiles = left_smiles.replace('*', '[*]')
            right_smiles = right_smiles.replace('*', '[*]')
            
            left_mol = Chem.MolFromSmiles(left_smiles)
            right_mol = Chem.MolFromSmiles(right_smiles)
            
            if left_mol is None or right_mol is None:
                return None
            
            # Create reaction
            rxn_smarts = f"{Chem.MolToSmarts(left_mol)}>>{Chem.MolToSmarts(right_mol)}"
            rxn = AllChem.ReactionFromSmarts(rxn_smarts)
            
            # Draw reaction
            img = Draw.ReactionToImage(rxn, subImgSize=(width, height))
            
            # Convert to base64
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode()
            
        except Exception as e:
            if show_debug:
                st.write(f"Reaction image error: {e}")
            return None
    
    def get_transform_description(transform):
        """Get human-readable description of transformation"""
        try:
            parts = transform.split('>>')
            if len(parts) == 2:
                left = parts[0]
                right = parts[1]
                
                # Remove brackets and attachment points
                left_clean = left.replace('[', '').replace(']', '').replace('*', '')
                right_clean = right.replace('[', '').replace(']', '').replace('*', '')
                
                # Common group mappings
                group_map = {
                    'CH3': 'methyl',
                    'CH2': 'methylene',
                    'CH': 'methine',
                    'NH2': 'amino',
                    'NH': 'imino',
                    'OH': 'hydroxyl',
                    'SH': 'sulfhydryl',
                    'F': 'fluoro',
                    'Cl': 'chloro',
                    'Br': 'bromo',
                    'I': 'iodo',
                    'CF3': 'trifluoromethyl',
                    'CN': 'cyano',
                    'NO2': 'nitro',
                    'CO2H': 'carboxy',
                    'CONH2': 'carbamoyl',
                    'C=O': 'carbonyl',
                    'O': 'oxy',
                    'N': 'aza',
                    'S': 'thia',
                }
                
                left_desc = group_map.get(left_clean, left_clean)
                right_desc = group_map.get(right_clean, right_clean)
                
                return f"{left_desc} → {right_desc}"
                
        except:
            pass
        return transform
    
    def create_download_link(data, filename, text):
        """Create a download link for data"""
        b64 = base64.b64encode(data.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
        return href

# Main app logic
if not RDKIT_AVAILABLE:
    st.error("""
    ## RDKit Not Available
    
    This app requires RDKit for MMPA functionality.
    
    **To install RDKit:**
    
    ### Option 1: Using pip
    ```bash
    pip install rdkit-pypi
    pip install "numpy<2"  # Important for compatibility
    ```
    
    ### Option 2: Using conda (recommended)
    ```bash
    conda install -c conda-forge rdkit
    ```
    """)
    
elif uploaded_file is not None:
    # Load data
    df = load_data(uploaded_file, sanitize=sanitize_molecules)
    
    if df is not None and len(df) > 0:
        # Show dataset info
        with st.expander("📊 Dataset Overview", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Compounds", len(df))
            col2.metric("Min pIC50", f"{df['pIC50'].min():.2f}")
            col3.metric("Max pIC50", f"{df['pIC50'].max():.2f}")
            col4.metric("Avg pIC50", f"{df['pIC50'].mean():.2f}")
            
            # Show sample data
            st.subheader("Sample Data (First 10 compounds)")
            st.dataframe(df[['Name', 'SMILES', 'pIC50']].head(10))
            
            # Show pIC50 distribution
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(df['pIC50'], bins=30, alpha=0.7, color='blue', edgecolor='black')
            ax.set_xlabel('pIC50')
            ax.set_ylabel('Frequency')
            ax.set_title('pIC50 Distribution')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        # Perform MMPA analysis
        st.markdown('<h2 class="section-header">🔬 RDKit MMPA Analysis</h2>', unsafe_allow_html=True)
        
        # Get parameters from sidebar
        min_occurrence_val = min_occurrence
        max_cuts_val = max_cuts
        max_cut_bonds_val = max_cut_bonds
        min_core_size_val = min_core_size
        min_delta_threshold_val = min_delta_threshold
        show_debug_val = show_debug_info
        
        # Perform analysis
        pairs_df, transforms_df, fragment_map = perform_rdkit_mmpa_analysis(
            df,
            max_cuts=max_cuts_val,
            max_cut_bonds=max_cut_bonds_val,
            min_transform_occurrence=min_occurrence_val,
            min_core_size=min_core_size_val,
            min_delta_threshold=min_delta_threshold_val,
            show_debug=show_debug_val
        )
        
        if pairs_df is not None:
            # Show analysis statistics
            st.success("✅ Analysis Complete!")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Pairs", len(pairs_df))
            
            if transforms_df is not None:
                col2.metric("Valid Transforms", len(transforms_df))
                col3.metric("Avg ΔpIC50", f"{pairs_df['Delta'].mean():.2f}")
                col4.metric("Significant Pairs", f"{(abs(pairs_df['Delta']) > 1.0).sum()}")
                
                # Show overall delta distribution
                st.markdown('<h3 class="section-header">📊 Overall ΔpIC50 Distribution</h3>', unsafe_allow_html=True)
                fig = plot_delta_distribution(pairs_df['Delta'].values, "All Molecular Pairs ΔpIC50")
                st.pyplot(fig)
                
                # Show top positive transforms
                if show_top_positive and len(transforms_df) > 0:
                    st.markdown('<h3 class="section-header">📈 Top Positive Transformations</h3>', unsafe_allow_html=True)
                    
                    top_n = min(5, len(transforms_df))
                    positive_transforms = transforms_df.head(top_n)
                    
                    for idx, (_, row) in enumerate(positive_transforms.iterrows()):
                        with st.container():
                            st.markdown(f"### Transform #{idx+1}")
                            
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                # Display reaction image
                                img_b64 = get_reaction_image(row['Transform'])
                                if img_b64:
                                    st.markdown(f'<img src="data:image/png;base64,{img_b64}" width="400">', unsafe_allow_html=True)
                                else:
                                    st.info("Reaction visualization not available")
                                
                                # Human-readable description
                                description = get_transform_description(row['Transform'])
                                st.markdown(f"**Description:** {description}")
                            
                            with col2:
                                # Display transform statistics
                                st.markdown(f"""
                                <div class="transform-card">
                                    <h4>Statistical Summary</h4>
                                    <p><strong>Transform:</strong> {row['Transform']}</p>
                                    <p><strong>Occurrences:</strong> {row['Count']}</p>
                                    <p><strong>Mean ΔpIC50:</strong> {row['Mean ΔpIC50']:.2f}</p>
                                    <p><strong>Median ΔpIC50:</strong> {row['Median ΔpIC50']:.2f}</p>
                                    <p><strong>Std ΔpIC50:</strong> {row['Std ΔpIC50']:.2f}</p>
                                    <p><strong>Range:</strong> {row['Min ΔpIC50']:.2f} to {row['Max ΔpIC50']:.2f}</p>
                                    <p><strong>Example:</strong> {row['Example_Names']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Plot delta distribution for this transform
                                fig = plot_delta_distribution(row['Deltas'], 
                                                            f"ΔpIC50 Distribution (n={row['Count']})")
                                st.pyplot(fig)
                            
                            # Show MCS analysis if requested
                            if show_mcs_analysis:
                                with st.expander("🔍 Maximum Common Substructure Analysis"):
                                    try:
                                        mol1 = Chem.MolFromSmiles(row['Example_SMILES_1'])
                                        mol2 = Chem.MolFromSmiles(row['Example_SMILES_2'])
                                        
                                        if mol1 and mol2:
                                            mcs_result = create_mcs_image(mol1, mol2)
                                            if mcs_result:
                                                img1_b64, img2_b64, mcs_smarts = mcs_result
                                                
                                                col1, col2 = st.columns(2)
                                                with col1:
                                                    st.markdown(f'<div class="highlight-before">Before: {row["Example_Names"].split("→")[0].strip()}</div>', unsafe_allow_html=True)
                                                    st.markdown(f'<img src="data:image/png;base64,{img1_b64}" width="400">', unsafe_allow_html=True)
                                                
                                                with col2:
                                                    st.markdown(f'<div class="highlight-after">After: {row["Example_Names"].split("→")[1].strip()}</div>', unsafe_allow_html=True)
                                                    st.markdown(f'<img src="data:image/png;base64,{img2_b64}" width="400">', unsafe_allow_html=True)
                                                
                                                st.markdown(f"**MCS SMARTS:** `{mcs_smarts}`")
                                                st.markdown(f"**MCS Size:** {mcs_smarts.count('[')} atoms")
                                            else:
                                                st.info("MCS analysis not available for this pair")
                                    except:
                                        st.info("MCS analysis failed for this pair")
                            
                            # Show compound examples
                            if show_compound_examples:
                                with st.expander(f"🧪 View Compound Examples ({row['Count']} pairs)"):
                                    # Filter pairs for this transform
                                    transform_pairs = pairs_df[pairs_df['Transform'] == row['Transform']]
                                    display_cols = ['Name_1', 'pIC50_1', 'Name_2', 'pIC50_2', 'Delta']
                                    st.dataframe(transform_pairs[display_cols].head(10))
                                    
                                    # Show molecule images for first few pairs
                                    st.subheader("Molecular Structures")
                                    for _, pair in transform_pairs.head(3).iterrows():
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            try:
                                                mol1 = Chem.MolFromSmiles(pair['SMILES_1'])
                                                if mol1:
                                                    img = Draw.MolToImage(mol1, size=(300, 300))
                                                    st.image(img, caption=f"{pair['Name_1']} (pIC50: {pair['pIC50_1']:.2f})")
                                            except:
                                                pass
                                        
                                        with col2:
                                            try:
                                                mol2 = Chem.MolFromSmiles(pair['SMILES_2'])
                                                if mol2:
                                                    img = Draw.MolToImage(mol2, size=(300, 300))
                                                    st.image(img, caption=f"{pair['Name_2']} (pIC50: {pair['pIC50_2']:.2f})")
                                            except:
                                                pass
                                        
                                        st.markdown(f"**ΔpIC50:** {pair['Delta']:.2f}")
                                        st.markdown("---")
                            
                            st.markdown("---")
                
                # Show top negative transforms
                if show_top_negative and len(transforms_df) > 0:
                    st.markdown('<h3 class="section-header">📉 Top Negative Transformations</h3>', unsafe_allow_html=True)
                    
                    top_n = min(5, len(transforms_df))
                    negative_transforms = transforms_df.tail(top_n).iloc[::-1]
                    
                    for idx, (_, row) in enumerate(negative_transforms.iterrows()):
                        with st.container():
                            st.markdown(f"### Transform #{idx+1} (Negative Impact)")
                            
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                img_b64 = get_reaction_image(row['Transform'])
                                if img_b64:
                                    st.markdown(f'<img src="data:image/png;base64,{img_b64}" width="400">', unsafe_allow_html=True)
                                
                                description = get_transform_description(row['Transform'])
                                st.markdown(f"**Description:** {description}")
                            
                            with col2:
                                st.markdown(f"""
                                <div class="transform-card">
                                    <h4>Statistical Summary</h4>
                                    <p><strong>Transform:</strong> {row['Transform']}</p>
                                    <p><strong>Occurrences:</strong> {row['Count']}</p>
                                    <p><strong>Mean ΔpIC50:</strong> {row['Mean ΔpIC50']:.2f}</p>
                                    <p><strong>Median ΔpIC50:</strong> {row['Median ΔpIC50']:.2f}</p>
                                    <p><strong>Std ΔpIC50:</strong> {row['Std ΔpIC50']:.2f}</p>
                                    <p><strong>Range:</strong> {row['Min ΔpIC50']:.2f} to {row['Max ΔpIC50']:.2f}</p>
                                    <p><strong>Example:</strong> {row['Example_Names']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                fig = plot_delta_distribution(row['Deltas'], 
                                                            f"ΔpIC50 Distribution (n={row['Count']})")
                                st.pyplot(fig)
                            
                            if show_compound_examples:
                                with st.expander(f"🧪 View Compound Examples"):
                                    transform_pairs = pairs_df[pairs_df['Transform'] == row['Transform']]
                                    display_cols = ['Name_1', 'pIC50_1', 'Name_2', 'pIC50_2', 'Delta']
                                    st.dataframe(transform_pairs[display_cols].head(10))
                            
                            st.markdown("---")
                
                # Show all transforms table
                if len(transforms_df) > 0:
                    st.markdown('<h3 class="section-header">📋 All Transformations</h3>', unsafe_allow_html=True)
                    
                    if show_all_transforms:
                        display_df = transforms_df
                    else:
                        display_df = transforms_df.head(transforms_to_display)
                    
                    # Create summary table
                    summary_cols = ['Transform', 'Count', 'Mean ΔpIC50', 'Std ΔpIC50', 
                                   'Min ΔpIC50', 'Max ΔpIC50', 'Example_Names']
                    st.dataframe(display_df[summary_cols])
                    
                    # Export option
                    if save_results:
                        st.markdown('<h3 class="section-header">💾 Export Results</h3>', unsafe_allow_html=True)
                        
                        # Convert DataFrames to CSV
                        pairs_csv = pairs_df.to_csv(index=False)
                        transforms_csv = transforms_df.to_csv(index=False)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(create_download_link(pairs_csv, "mmp_pairs.csv", 
                                                           "📥 Download All Pairs (CSV)"), 
                                      unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(create_download_link(transforms_csv, "mmp_transforms.csv", 
                                                           "📥 Download Transforms (CSV)"), 
                                      unsafe_allow_html=True)
                        
                        # Excel export
                        try:
                            import openpyxl
                            
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                pairs_df.to_excel(writer, sheet_name='All_Pairs', index=False)
                                transforms_df.to_excel(writer, sheet_name='Transforms', index=False)
                                df.to_excel(writer, sheet_name='Input_Data', index=False)
                            
                            excel_data = output.getvalue()
                            st.download_button(
                                label="📥 Download Full Analysis (Excel)",
                                data=excel_data,
                                file_name="mmpa_analysis.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        except:
                            st.info("Excel export requires openpyxl: pip install openpyxl")
            
            else:
                st.info(f"Found {len(pairs_df)} pairs but no transforms met the minimum occurrence ({min_occurrence_val})")
                
                # Show pair statistics
                with st.expander("View Generated Pairs"):
                    st.dataframe(pairs_df.head(20))
                    
                    fig = plot_delta_distribution(pairs_df['Delta'].values, "Generated Pairs ΔpIC50")
                    st.pyplot(fig)
    else:
        st.error("No valid data loaded. Please check your CSV file format.")
else:
    # Welcome message
    st.markdown("""
    ## Welcome to the RDKit MMPA Analysis Tool
    
    ### Features:
    - **Professional MMP Analysis**: Using RDKit's built-in MMPA algorithms
    - **Configurable Fragmentation**: Control bond cutting parameters
    - **Statistical Analysis**: Mean, median, standard deviation of ΔpIC50
    - **MCS Visualization**: Maximum Common Substructure highlighting
    - **Export Capabilities**: CSV and Excel export
    
    ### How to Use:
    1. **Upload a CSV file** with SMILES and pIC50 columns
    2. **Configure parameters** in the sidebar
    3. **View results** including top positive/negative transforms
    4. **Export findings** for further analysis
    
    ### Required CSV Format:
    Your CSV file should contain:
    - `SMILES`: Molecular structures
    - `pIC50`: Biological activity values
    - `Name`: Compound names (optional, will be auto-generated if missing)
    
    ### Example CSV:
    ```csv
    SMILES,pIC50,Name
    CC(=O)OC1=CC=CC=C1C(=O)O,4.5,Aspirin
    CN1C=NC2=C1C(=O)N(C(=O)N2C)C,5.2,Caffeine
    ```
    
    ⬅️ **Upload a CSV file in the sidebar to begin analysis**
    """)
    
    # Example data
    with st.expander("📁 Download Example Dataset"):
        st.markdown("""
        Download this example dataset to test the tool:
        
        [Example_MMPA_Dataset.csv](https://raw.githubusercontent.com/rdkit/rdkit/master/Contrib/mmpa/example.csv)
        
        Or create your own with this template:
        """)
        
        example_data = {
            'SMILES': ['CC(=O)OC1=CC=CC=C1C(=O)O', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C', 
                      'C1=CC=C(C=C1)C=O', 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O'],
            'pIC50': [4.5, 5.2, 3.8, 6.1],
            'Name': ['Aspirin', 'Caffeine', 'Benzaldehyde', 'Ibuprofen']
        }
        
        example_df = pd.DataFrame(example_data)
        st.dataframe(example_df)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; font-size: 0.9rem;">
    <p>RDKit MMPA Analysis Tool v2.0 | Built with RDKit's official MMPA implementation</p>
    <p>For research use only. Cite: Hussain & Rea, J. Chem. Inf. Model., 2010</p>
    <p>RDKit Documentation: https://www.rdkit.org/docs/Cookbook.html#mmpa</p>
</div>
""", unsafe_allow_html=True)
