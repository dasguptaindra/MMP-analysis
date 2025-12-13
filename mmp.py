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
    .info-box {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
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
        min_occurrence = st.slider("Minimum transform occurrences", 1, 20, 5, 
                                  help="Minimum number of occurrences for a transform to be considered")
        
        # CUTS PARAMETER - MODIFIED
        st.markdown("### ✂️ Fragmentation Options")
        max_cuts = st.selectbox(
            "Maximum number of cuts per molecule",
            options=[1, 2, 3, 4, 5],
            index=0,
            help="Maximum number of bonds to break to generate fragments. Higher values create more complex transformations."
        )
        
        # Additional fragmentation options
        st.markdown("#### Advanced Fragmentation")
        
        fragmentation_strategy = st.selectbox(
            "Fragmentation strategy",
            options=["Single cuts only", "All cuts up to max", "Smart fragmentation"],
            index=0,
            help="""Single cuts only: Break only one bond per molecule
                   All cuts up to max: Try all combinations up to max cuts
                   Smart fragmentation: Only cut at rotatable bonds"""
        )
        
        include_rings = st.checkbox(
            "Allow cutting ring bonds", 
            value=False,
            help="Allow fragmentation that breaks ring structures (may create unrealistic transformations)"
        )
        
        # Molecule cleaning options
        st.markdown("### 🧹 Molecule Cleaning")
        sanitize_molecules = st.checkbox("Sanitize molecules", value=True,
                                       help="Clean molecules (recommended)")
        kekulize_molecules = st.checkbox("Kekulize molecules", value=False,
                                        help="Force kekulization (may fail for some molecules)")
        
        # Display options
        st.markdown("### 👀 Display Options")
        show_all_transforms = st.checkbox("Show all transformations", value=False)
        transforms_to_display = st.slider("Number of transforms to display", 1, 50, 10, 
                                         disabled=show_all_transforms)
        
        # Analysis options
        st.markdown("### 🔬 Analysis")
        show_top_positive = st.checkbox("Show top positive transforms", value=True)
        show_top_negative = st.checkbox("Show top negative transforms", value=True)
        show_compound_examples = st.checkbox("Show compound examples", value=True)
        
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
    
    # About section
    with st.expander("About MMP Analysis"):
        st.markdown("""
        **Matched Molecular Pairs (MMP)** is a technique for identifying structural transformations that affect biological activity.
        
        **Key steps:**
        1. Decompose molecules into core and R-groups
        2. Find pairs with same core but different R-groups
        3. Calculate ΔpIC50 for each pair
        4. Identify frequently occurring transformations
        
        **Number of cuts** controls how many bonds are broken:
        - **1 cut**: Simple R-group replacements (most common)
        - **2-3 cuts**: More complex scaffold hopping
        - **4+ cuts**: Major structural changes
        """)

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

    def sort_fragments(mol):
        """Sort fragments by number of atoms"""
        if mol is None:
            return []
        try:
            frag_list = list(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False))
            frag_list = [remove_map_nums(x) for x in frag_list]
            frag_num_atoms_list = [(x.GetNumAtoms(), x) for x in frag_list]
            frag_num_atoms_list.sort(key=itemgetter(0), reverse=True)
            return [x[1] for x in frag_num_atoms_list]
        except Exception as e:
            return []

    def FragmentMol(mol, maxCuts=1, strategy="single", include_rings=False):
        """Fragmentation function with configurable cuts"""
        results = []
        try:
            # Create a copy to avoid modifying original
            mol_copy = Chem.Mol(mol)
            
            # Get all bonds that can be cut
            bonds_to_consider = []
            for bond in mol_copy.GetBonds():
                # Check bond type
                if bond.GetBondType() == Chem.BondType.SINGLE:
                    # Check if it's a ring bond
                    is_ring_bond = bond.IsInRing()
                    
                    # Apply ring bond filtering
                    if not include_rings and is_ring_bond:
                        continue
                    
                    bonds_to_consider.append(bond)
            
            if strategy == "single":
                # Single cuts only - original behavior
                for bond in bonds_to_consider:
                    try:
                        emol = Chem.EditableMol(mol_copy)
                        emol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                        frag_mol = emol.GetMol()
                        
                        try:
                            Chem.SanitizeMol(frag_mol)
                        except:
                            pass
                        
                        results.append((f"CUT_{bond.GetIdx()}", frag_mol))
                    except:
                        continue
            
            elif strategy == "all":
                # All combinations up to maxCuts
                for num_cuts in range(1, min(maxCuts, len(bonds_to_consider)) + 1):
                    # Generate combinations of bonds to cut
                    bond_combinations = list(combinations(bonds_to_consider, num_cuts))
                    
                    for bond_combo in bond_combinations:
                        try:
                            emol = Chem.EditableMol(mol_copy)
                            bond_indices = []
                            
                            # Remove all bonds in the combination
                            for bond in bond_combo:
                                emol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                                bond_indices.append(bond.GetIdx())
                            
                            frag_mol = emol.GetMol()
                            
                            try:
                                Chem.SanitizeMol(frag_mol)
                            except:
                                pass
                            
                            # Only keep if we have at least 2 fragments
                            frag_list = sort_fragments(frag_mol)
                            if len(frag_list) >= 2:
                                combo_str = f"CUTS_{'_'.join(map(str, sorted(bond_indices)))}"
                                results.append((combo_str, frag_mol))
                        except:
                            continue
            
            elif strategy == "smart":
                # Smart fragmentation - only cut at rotatable bonds
                from rdkit.Chem import rdMolDescriptors
                
                # Get rotatable bonds
                rotatable_bonds = []
                for bond in bonds_to_consider:
                    # Simple rotatable bond definition (can be improved)
                    if not bond.IsInRing():
                        # Check if both atoms are not in terminal groups
                        begin_atom = bond.GetBeginAtom()
                        end_atom = bond.GetEndAtom()
                        
                        if (begin_atom.GetDegree() > 1 and end_atom.GetDegree() > 1):
                            rotatable_bonds.append(bond)
                
                # Try single cuts on rotatable bonds
                for bond in rotatable_bonds:
                    try:
                        emol = Chem.EditableMol(mol_copy)
                        emol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                        frag_mol = emol.GetMol()
                        
                        try:
                            Chem.SanitizeMol(frag_mol)
                        except:
                            pass
                        
                        results.append((f"ROT_CUT_{bond.GetIdx()}", frag_mol))
                    except:
                        continue
            
            # If no cuts were made, add the original molecule
            if not results:
                results.append(("NO_CUT", mol))
            
            return results
            
        except Exception as e:
            # Return at least the original molecule
            return [("ERROR", mol)]

    def perform_mmp_analysis(df, min_transform_occurrence, max_cuts=1, 
                           fragmentation_strategy="single", include_rings=False,
                           show_debug=False):
        """Perform MMP analysis with configurable cuts"""
        if df is None or len(df) == 0:
            return None, None
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Map strategy string to function parameter
        strategy_map = {
            "Single cuts only": "single",
            "All cuts up to max": "all",
            "Smart fragmentation": "smart"
        }
        strategy = strategy_map.get(fragmentation_strategy, "single")
        
        # Step 1: Decompose molecules
        status_text.text(f"Step 1/4: Decomposing molecules (max cuts={max_cuts})...")
        progress_bar.progress(25)
        
        row_list = []
        successful = 0
        failed = 0
        fragmentation_stats = {"single_cuts": 0, "multiple_cuts": 0}
        
        for idx, row in df.iterrows():
            smiles = row['SMILES']
            name = row.get('Name', f"CMPD_{idx}")
            pIC50 = row['pIC50']
            mol = row['mol']
            
            if mol is None:
                failed += 1
                continue
                
            try:
                # Use configurable fragmentation
                frag_list = FragmentMol(mol, maxCuts=max_cuts, 
                                      strategy=strategy, include_rings=include_rings)
                
                for frag_name, frag_mol in frag_list:
                    pair_list = sort_fragments(frag_mol)
                    if len(pair_list) >= 2:
                        # Count fragmentation type
                        if "CUTS_" in frag_name:
                            fragmentation_stats["multiple_cuts"] += 1
                        else:
                            fragmentation_stats["single_cuts"] += 1
                        
                        # Convert to SMILES with error handling
                        try:
                            core_smiles = Chem.MolToSmiles(pair_list[0])
                            rgroup_smiles = Chem.MolToSmiles(pair_list[1])
                            tmp_list = [smiles, core_smiles, rgroup_smiles, name, pIC50, frag_name]
                            row_list.append(tmp_list)
                            successful += 1
                        except:
                            failed += 1
            except Exception as e:
                failed += 1
                continue
        
        if not row_list:
            st.error("No valid fragments found")
            return None, None
        
        # Create DataFrame
        row_df = pd.DataFrame(row_list, columns=["SMILES", "Core", "R_group", "Name", "pIC50", "Fragmentation_Type"])
        
        # Show fragmentation statistics
        st.markdown(f"""
        <div class="info-box">
            <h4>📊 Fragmentation Statistics</h4>
            <p>• Total valid fragments: {successful}</p>
            <p>• Single-cut fragments: {fragmentation_stats['single_cuts']}</p>
            <p>• Multiple-cut fragments: {fragmentation_stats['multiple_cuts']}</p>
            <p>• Unique cores identified: {row_df['Core'].nunique()}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if show_debug:
            with st.expander("Debug: Row DataFrame", expanded=False):
                st.write(f"Total rows: {len(row_df)}")
                st.dataframe(row_df.head(20))
                st.write("Fragmentation types distribution:")
                st.write(row_df['Fragmentation_Type'].value_counts())
        
        if failed > 0:
            st.info(f"Successfully processed {successful} molecules, failed on {failed}")
        
        # Step 2: Collect pairs
        status_text.text("Step 2/4: Collecting molecular pairs...")
        progress_bar.progress(50)
        
        delta_list = []
        cores_with_pairs = 0
        total_combinations = 0
        
        # Group by Core and iterate through each group
        for k, v in row_df.groupby("Core"):
            # Only process groups with more than 2 compounds
            if len(v) > 2:
                cores_with_pairs += 1
                # Generate all unique combinations of indices
                combinations_list = list(combinations(range(0, len(v)), 2))
                total_combinations += len(combinations_list)
                
                for a, b in combinations_list:
                    reagent_a = v.iloc[a]
                    reagent_b = v.iloc[b]
                    
                    # Skip if same molecule
                    if reagent_a.SMILES == reagent_b.SMILES:
                        continue
                    
                    # Sort by SMILES for canonical ordering
                    reagent_a, reagent_b = sorted([reagent_a, reagent_b], key=lambda x: x.SMILES)
                    
                    # Calculate delta
                    delta = reagent_b.pIC50 - reagent_a.pIC50
                    
                    # Create transform string
                    transform_str = f"{reagent_a.R_group.replace('*','*-')}>>{reagent_b.R_group.replace('*','*-')}"
                    
                    # Store fragmentation type in transform name
                    frag_type_a = reagent_a.Fragmentation_Type
                    frag_type_b = reagent_b.Fragmentation_Type
                    frag_types = f"{frag_type_a}|{frag_type_b}"
                    
                    delta_list.append([
                        reagent_a.SMILES, reagent_a.Core, reagent_a.R_group, reagent_a.Name, reagent_a.pIC50, frag_type_a,
                        reagent_b.SMILES, reagent_b.Core, reagent_b.R_group, reagent_b.Name, reagent_b.pIC50, frag_type_b,
                        transform_str, delta, frag_types
                    ])
        
        if show_debug:
            with st.expander("Debug: Pair Generation", expanded=False):
                st.write(f"Cores with >2 compounds: {cores_with_pairs}")
                st.write(f"Total possible combinations: {total_combinations}")
                st.write(f"Actual pairs generated: {len(delta_list)}")
        
        if not delta_list:
            st.error("No molecular pairs found")
            return None, None
        
        # Create DataFrame
        cols = [
            "SMILES_1", "Core_1", "R_group_1", "Name_1", "pIC50_1", "Frag_Type_1",
            "SMILES_2", "Core_2", "R_group_2", "Name_2", "pIC50_2", "Frag_Type_2",
            "Transform", "Delta", "Fragmentation_Types"
        ]
        delta_df = pd.DataFrame(delta_list, columns=cols)
        
        if show_debug:
            with st.expander("Debug: Delta DataFrame", expanded=False):
                st.write(f"Delta DataFrame shape: {delta_df.shape}")
                st.dataframe(delta_df.head(10))
        
        # Step 3: Collect frequent transforms
        status_text.text("Step 3/4: Analyzing transformations...")
        progress_bar.progress(75)
        
        mmp_list = []
        for k, v in delta_df.groupby("Transform"):
            # Only include transforms with minimum occurrences
            if len(v) >= min_transform_occurrence:
                # Get fragmentation types for this transform
                frag_types = v['Fragmentation_Types'].unique()
                mmp_list.append([k, len(v), v.Delta.values, frag_types])
        
        if show_debug:
            with st.expander("Debug: Transform Collection", expanded=False):
                st.write(f"Total unique transforms: {delta_df['Transform'].nunique()}")
                st.write(f"Transforms with >= {min_transform_occurrence} occurrences: {len(mmp_list)}")
        
        if not mmp_list:
            st.warning(f"No transforms found with {min_transform_occurrence}+ occurrences")
            return delta_df, None
        
        # Create transforms DataFrame
        mmp_df = pd.DataFrame(mmp_list, columns=["Transform", "Count", "Deltas", "Fragmentation_Types"])
        mmp_df['idx'] = range(0, len(mmp_df))
        mmp_df['mean_delta'] = [x.mean() for x in mmp_df.Deltas]
        mmp_df['std_delta'] = [x.std() for x in mmp_df.Deltas]
        mmp_df['min_delta'] = [x.min() for x in mmp_df.Deltas]
        mmp_df['max_delta'] = [x.max() for x in mmp_df.Deltas]
        
        # Create reaction molecules with error handling
        rxn_mols = []
        for transform in mmp_df['Transform']:
            try:
                rxn = AllChem.ReactionFromSmarts(transform.replace('*-','*'), useSmiles=True)
                rxn_mols.append(rxn)
            except Exception as e:
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
        
        # Add fragmentation complexity score
        def get_frag_complexity(frag_types_array):
            """Calculate fragmentation complexity score"""
            complexity = 0
            for types in frag_types_array:
                for t in types.split('|'):
                    if 'CUTS_' in t:
                        # Count number of cuts in CUTS_1_2_3 format
                        cut_count = len(t.split('_')) - 1
                        complexity = max(complexity, cut_count)
                    elif t != 'NO_CUT':
                        complexity = max(complexity, 1)
            return complexity
        
        mmp_df['frag_complexity'] = [get_frag_complexity(x) for x in mmp_df['Fragmentation_Types']]
        
        # Step 4: Complete
        status_text.text("Step 4/4: Analysis complete!")
        progress_bar.progress(100)
        status_text.empty()
        progress_bar.empty()
        
        return delta_df, mmp_df

    def plot_stripplot_to_fig(deltas):
        """Create a stripplot figure"""
        fig, ax = plt.subplots(figsize=(4, 1.5))
        sns.stripplot(x=deltas, ax=ax, jitter=0.2, alpha=0.7, s=5, color='blue')
        ax.axvline(0, ls='--', c='red')
        
        # Set appropriate x limits based on data
        if len(deltas) > 0:
            data_min = min(deltas)
            data_max = max(deltas)
            padding = max(0.5, (data_max - data_min) * 0.1)
            ax.set_xlim(data_min - padding, data_max + padding)
        else:
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
                "Frag_Type": row['Frag_Type_1'],
                "Type": "Before"
            })
            example_list.append({
                "SMILES": row['SMILES_2'],
                "Name": row['Name_2'],
                "pIC50": row['pIC50_2'],
                "Frag_Type": row['Frag_Type_2'],
                "Type": "After"
            })
        
        return pd.DataFrame(example_list)

    def display_molecule_grid(smiles_list, names_list, pIC50_list):
        """Display molecules in a grid"""
        cols = st.columns(4)
        for idx, (smiles, name, pIC50) in enumerate(zip(smiles_list, names_list, pIC50_list)):
            with cols[idx % 4]:
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        img = Draw.MolToImage(mol, size=(200, 200))
                        st.image(img, caption=f"{name} (pIC50: {pIC50:.2f})")
                except:
                    st.write(f"{name}: {smiles}")

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
            
            st.dataframe(df[['SMILES', 'Name', 'pIC50']].head(10))
        
        # Perform MMP analysis with configurable cuts
        st.markdown('<h2 class="section-header">🔍 MMP Analysis Results</h2>', unsafe_allow_html=True)
        
        # Show analysis parameters
        st.markdown(f"""
        <div class="info-box">
            <h4>⚙️ Analysis Parameters</h4>
            <p>• Maximum cuts per molecule: <strong>{max_cuts}</strong></p>
            <p>• Fragmentation strategy: <strong>{fragmentation_strategy}</strong></p>
            <p>• Allow ring bond cutting: <strong>{'Yes' if include_rings else 'No'}</strong></p>
            <p>• Minimum transform occurrences: <strong>{min_occurrence}</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        delta_df, mmp_df = perform_mmp_analysis(
            df, 
            min_occurrence, 
            max_cuts=max_cuts,
            fragmentation_strategy=fragmentation_strategy,
            include_rings=include_rings,
            show_debug=show_debug
        )
        
        if delta_df is not None:
            # Show statistics
            st.success("Analysis complete!")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Pairs Generated", len(delta_df))
            
            if mmp_df is not None:
                col2.metric("Unique Transforms", len(mmp_df))
                col3.metric("Avg Transform Frequency", f"{mmp_df['Count'].mean():.1f}")
                
                # Calculate fragmentation complexity distribution
                complexity_counts = mmp_df['frag_complexity'].value_counts().sort_index()
                complexity_summary = ", ".join([f"{k} cuts: {v}" for k, v in complexity_counts.items()])
                col4.metric("Fragmentation Complexity", complexity_summary[:30] + "..." if len(complexity_summary) > 30 else complexity_summary)
                
                # Sort transforms by mean delta
                mmp_df_sorted = mmp_df.sort_values("mean_delta", ascending=False)
                
                # Show top positive transforms
                if show_top_positive and len(mmp_df_sorted) > 0:
                    st.markdown('<h3 class="section-header">📈 Top Positive Transformations</h3>', unsafe_allow_html=True)
                    top_positive = mmp_df_sorted.head(min(3, len(mmp_df_sorted)))
                    
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
                                # Show fragmentation complexity badge
                                complexity_badge = ""
                                if row['frag_complexity'] > 1:
                                    complexity_badge = f'<span style="background-color: #F59E0B; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8rem;">{row["frag_complexity"]} cuts</span>'
                                
                                st.markdown(f"""
                                <div class="transform-card">
                                    <h4>Transform #{i+1} {complexity_badge}</h4>
                                    <p><strong>Transformation:</strong> {row['Transform']}</p>
                                    <p><strong>Mean ΔpIC50:</strong> {row['mean_delta']:.2f} ± {row['std_delta']:.2f}</p>
                                    <p><strong>Occurrences:</strong> {row['Count']}</p>
                                    <p><strong>ΔpIC50 Range:</strong> {row['min_delta']:.2f} to {row['max_delta']:.2f}</p>
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
                                        # Display as table first
                                        st.dataframe(examples_df)
                
                # Show top negative transforms
                if show_top_negative and len(mmp_df_sorted) > 0:
                    st.markdown('<h3 class="section-header">📉 Top Negative Transformations</h3>', unsafe_allow_html=True)
                    top_negative = mmp_df_sorted.tail(min(3, len(mmp_df_sorted))).iloc[::-1]
                    
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
                                # Show fragmentation complexity badge
                                complexity_badge = ""
                                if row['frag_complexity'] > 1:
                                    complexity_badge = f'<span style="background-color: #F59E0B; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8rem;">{row["frag_complexity"]} cuts</span>'
                                
                                st.markdown(f"""
                                <div class="transform-card">
                                    <h4>Transform #{i+1} (Negative) {complexity_badge}</h4>
                                    <p><strong>Transformation:</strong> {row['Transform']}</p>
                                    <p><strong>Mean ΔpIC50:</strong> {row['mean_delta']:.2f} ± {row['std_delta']:.2f}</p>
                                    <p><strong>Occurrences:</strong> {row['Count']}</p>
                                    <p><strong>ΔpIC50 Range:</strong> {row['min_delta']:.2f} to {row['max_delta']:.2f}</p>
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
                                        # Display as table first
                                        st.dataframe(examples_df)
                
                # Show all transforms table with filtering options
                if len(mmp_df_sorted) > 0:
                    st.markdown('<h3 class="section-header">📋 All Transformations</h3>', unsafe_allow_html=True)
                    
                    # Add filtering options
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        filter_complexity = st.selectbox(
                            "Filter by fragmentation complexity",
                            options=["All", "Single cuts only", "Multiple cuts only"],
                            index=0
                        )
                    
                    with col2:
                        filter_direction = st.selectbox(
                            "Filter by effect direction",
                            options=["All", "Positive only (Δ>0)", "Negative only (Δ<0)"],
                            index=0
                        )
                    
                    with col3:
                        min_frequency = st.slider(
                            "Minimum frequency",
                            min_value=min_occurrence,
                            max_value=int(mmp_df_sorted['Count'].max()),
                            value=min_occurrence
                        )
                    
                    # Apply filters
                    filtered_df = mmp_df_sorted.copy()
                    
                    if filter_complexity == "Single cuts only":
                        filtered_df = filtered_df[filtered_df['frag_complexity'] == 1]
                    elif filter_complexity == "Multiple cuts only":
                        filtered_df = filtered_df[filtered_df['frag_complexity'] > 1]
                    
                    if filter_direction == "Positive only (Δ>0)":
                        filtered_df = filtered_df[filtered_df['mean_delta'] > 0]
                    elif filter_direction == "Negative only (Δ<0)":
                        filtered_df = filtered_df[filtered_df['mean_delta'] < 0]
                    
                    filtered_df = filtered_df[filtered_df['Count'] >= min_frequency]
                    
                    st.info(f"Showing {len(filtered_df)} transforms after filtering")
                    
                    if show_all_transforms:
                        display_df = filtered_df
                    else:
                        display_df = filtered_df.head(transforms_to_display)
                    
                    # Enhanced table display
                    display_columns = ['Transform', 'Count', 'mean_delta', 'std_delta', 'frag_complexity']
                    st.dataframe(display_df[display_columns].rename(
                        columns={
                            'mean_delta': 'Mean ΔpIC50',
                            'std_delta': 'Std ΔpIC50',
                            'frag_complexity': 'Cuts'
                        }
                    ).round(3))
                
                # Export results
                if save_results and mmp_df is not None:
                    st.markdown('<h3 class="section-header">💾 Export Results</h3>', unsafe_allow_html=True)
                    
                    # Create downloadable files
                    @st.cache_data
                    def convert_df_to_csv(df):
                        return df.to_csv(index=False).encode('utf-8')
                    
                    @st.cache_data
                    def convert_df_to_excel(df):
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            df.to_excel(writer, index=False, sheet_name='MMP_Results')
                            delta_df.to_excel(writer, index=False, sheet_name='All_Pairs')
                        return output.getvalue()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.download_button(
                            label="📥 Download MMP Results (CSV)",
                            data=convert_df_to_csv(mmp_df_sorted),
                            file_name=f"mmp_results_{max_cuts}cuts.csv",
                            mime="text/csv"
                        )
                    
                    with col2:
                        st.download_button(
                            label="📥 Download MMP Results (Excel)",
                            data=convert_df_to_excel(mmp_df_sorted),
                            file_name=f"mmp_results_{max_cuts}cuts.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    
                    # Also provide delta pairs
                    st.download_button(
                        label="📥 Download All Molecular Pairs (CSV)",
                        data=convert_df_to_csv(delta_df),
                        file_name=f"mmp_pairs_{max_cuts}cuts.csv",
                        mime="text/csv"
                    )
            
            else:
                st.info(f"No transformations found with {min_occurrence}+ occurrences. Try reducing the minimum occurrence threshold or increasing the number of cuts.")
    else:
        st.warning("No valid molecules found in the dataset. Please check your SMILES strings.")
else:
    # Show welcome message when no file is uploaded
    st.markdown("""
    ## Welcome to the MMP Analysis Tool! 👋
    
    This tool performs **Matched Molecular Pair (MMP) analysis** to identify structural transformations that affect compound potency.
    
    ### Feature: Configurable Number of Cuts
    
    You can now choose how many bonds to break when fragmenting molecules:
    
    - **1 cut**: Simple R-group replacements (traditional MMP)
    - **2-3 cuts**: Scaffold hopping and linker modifications
    - **4+ cuts**: Major structural changes and core modifications
    
    ### How to use:
    1. **Upload your data** using the sidebar on the left
    2. **Configure fragmentation parameters** (number of cuts, strategy)
    3. **Set analysis parameters** like minimum transform occurrences
    4. **View results** including top positive/negative transformations
    5. **Export findings** for further analysis
    
    ### Required CSV format:
    Your CSV file should contain at least these columns:
    - `SMILES`: Molecular structures in SMILES format
    - `pIC50`: Potency values (negative log of IC50)
    - `Name`: Compound names (optional but recommended)
    
    ### Example CSV format:
    ```csv
    SMILES,Name,pIC50
    CC(=O)OC1=CC=CC=C1C(=O)O,aspirin,5.0
    CN1C=NC2=C1C(=O)N(C(=O)N2C)C,caffeine,4.2
    C1=CC=C(C=C1)C=O,benzaldehyde,3.8
    ...
    ```
    
    ### Key Logic:
    - **Pairs are generated only when 3+ compounds share the same core**
    - **Configurable cuts** allow you to explore different levels of structural changes
    - **Multiple fragmentation strategies** available for different analysis needs
    
    ### Troubleshooting:
    If you encounter errors:
    1. **NumPy compatibility**: Install `numpy<2` with `pip install "numpy<2"`
    2. **Invalid SMILES**: Check your SMILES strings are valid
    3. **Memory issues with many cuts**: Reduce max cuts or use smarter fragmentation
    
    ⬅️ **Upload a CSV file in the sidebar to get started!**
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; font-size: 0.9rem;">
    <p>MMP Analysis Tool v2.0 | Configurable Cuts Edition | Built with Streamlit, RDKit, and Pandas</p>
    <p>For research use only. Always validate computational predictions with experimental data.</p>
</div>
""", unsafe_allow_html=True)

