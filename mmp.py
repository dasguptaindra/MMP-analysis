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
    from rdkit.Chem import AllChem, Draw, rdMolDescriptors
    from rdkit.Chem.Draw import rdMolDraw2D
    # Try to import RDKit MMPA module
    try:
        from rdkit.Chem.rdMMPA import FragmentMol as RDKitFragmentMol
        RDKIT_MMPA_AVAILABLE = True
    except ImportError:
        RDKIT_MMPA_AVAILABLE = False
        st.warning("RDKit MMPA module not available. Using legacy fragmentation method.")
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
        
        # Fragmentation Method Selection
        st.markdown("### ✂️ Fragmentation Method")
        
        fragmentation_method = st.selectbox(
            "Fragmentation algorithm",
            options=["RDKit MMPA (Recommended)", "Legacy Method", "Auto-select"],
            index=0,
            help="""RDKit MMPA: Uses RDKit's optimized fragmentation algorithm (fastest)
                   Legacy Method: Original bond-breaking method (more control)
                   Auto-select: Automatically chooses the best method"""
        )
        
        # CUTS PARAMETER - Only show for RDKit MMPA
        if fragmentation_method != "Legacy Method" and RDKIT_MMPA_AVAILABLE:
            max_cuts = st.selectbox(
                "Maximum number of cuts",
                options=[1, 2, 3, 4],
                index=0,
                help="Maximum number of cuts for RDKit MMPA algorithm"
            )
        else:
            max_cuts = st.selectbox(
                "Maximum number of cuts",
                options=[1, 2, 3, 4, 5],
                index=0,
                help="Maximum number of bonds to break (Legacy method)"
            )
        
        # Advanced fragmentation options
        st.markdown("#### Advanced Fragmentation")
        
        if fragmentation_method == "Legacy Method":
            fragmentation_strategy = st.selectbox(
                "Fragmentation strategy",
                options=["Single cuts only", "All cuts up to max", "Smart fragmentation"],
                index=0,
                help="""Single cuts only: Break only one bond per molecule
                       All cuts up to max: Try all combinations up to max cuts
                       Smart fragmentation: Only cut at rotatable bonds"""
            )
        else:
            fragmentation_strategy = "RDKit MMPA"
        
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
        
        **Fragmentation Methods:**
        - **RDKit MMPA**: Optimized algorithm for efficient fragmentation
        - **Legacy Method**: Manual bond breaking with full control
        - **Number of cuts** controls fragmentation complexity
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

    def validate_and_clean_fragments(frag_mol, min_atoms=3, max_ratio=10):
        """Validate and clean fragmentation results"""
        try:
            frag_list = sort_fragments(frag_mol)
            if len(frag_list) >= 2:
                # Get the two largest fragments
                main_frags = frag_list[:2]
                min_frag_size = min(f.GetNumAtoms() for f in main_frags)
                max_frag_size = max(f.GetNumAtoms() for f in main_frags)
                
                # Avoid extremely small fragments and extreme size ratios
                if min_frag_size >= min_atoms and max_frag_size / min_frag_size < max_ratio:
                    return True
            return False
        except:
            return False

    def fragment_mol_legacy(mol, maxCuts=1, strategy="single", include_rings=False):
        """Legacy fragmentation method"""
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
            
            if strategy == "single" or maxCuts == 1:
                # Single cuts only
                for bond in bonds_to_consider:
                    try:
                        emol = Chem.EditableMol(mol_copy)
                        emol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                        frag_mol = emol.GetMol()
                        
                        try:
                            Chem.SanitizeMol(frag_mol)
                        except:
                            pass
                        
                        if validate_and_clean_fragments(frag_mol):
                            results.append((f"LEGACY_CUT_{bond.GetIdx()}", frag_mol))
                    except:
                        continue
            
            elif strategy == "all":
                # All combinations up to maxCuts
                for num_cuts in range(1, min(maxCuts, len(bonds_to_consider)) + 1):
                    bond_combinations = list(combinations(bonds_to_consider, num_cuts))
                    
                    for bond_combo in bond_combinations:
                        try:
                            emol = Chem.EditableMol(mol_copy)
                            bond_indices = []
                            
                            for bond in bond_combo:
                                emol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                                bond_indices.append(bond.GetIdx())
                            
                            frag_mol = emol.GetMol()
                            
                            try:
                                Chem.SanitizeMol(frag_mol)
                            except:
                                pass
                            
                            if validate_and_clean_fragments(frag_mol):
                                combo_str = f"LEGACY_CUTS_{'_'.join(map(str, sorted(bond_indices)))}"
                                results.append((combo_str, frag_mol))
                        except:
                            continue
            
            elif strategy == "smart":
                # Smart fragmentation - only cut at rotatable bonds
                rotatable_bonds = []
                for bond in bonds_to_consider:
                    if not bond.IsInRing():
                        begin_atom = bond.GetBeginAtom()
                        end_atom = bond.GetEndAtom()
                        
                        if (begin_atom.GetDegree() > 1 and end_atom.GetDegree() > 1):
                            # Simple rotatable bond check
                            if not (begin_atom.IsInRing() and end_atom.IsInRing()):
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
                        
                        if validate_and_clean_fragments(frag_mol):
                            results.append((f"SMART_CUT_{bond.GetIdx()}", frag_mol))
                    except:
                        continue
            
            # If no cuts were made, add the original molecule
            if not results:
                results.append(("NO_CUT", mol))
            
            return results
            
        except Exception as e:
            return [("ERROR", mol)]

    def FragmentMol(mol, maxCuts=1, strategy="single", include_rings=False):
        """Improved fragmentation using RDKit's MMPA module when available"""
        results = []
        
        # Determine which fragmentation method to use
        use_rdkit_mmpa = False
        if RDKIT_MMPA_AVAILABLE and strategy != "Legacy Method":
            if strategy == "RDKit MMPA" or (strategy == "Auto-select" and maxCuts <= 4):
                use_rdkit_mmpa = True
        
        if use_rdkit_mmpa:
            # Use RDKit MMPA
            try:
                # Adjust maxCuts for RDKit MMPA (typically 1-4)
                actual_max_cuts = min(maxCuts, 4)
                
                # Get fragments using RDKit MMPA
                fragments = RDKitFragmentMol(mol, maxCuts=actual_max_cuts)
                
                # Process fragments
                for frag_set in fragments:
                    if len(frag_set) >= 2:
                        try:
                            # Combine fragments into a single molecule for display
                            combined = Chem.Mol(frag_set[0])
                            for frag in frag_set[1:]:
                                combined = Chem.CombineMols(combined, frag)
                            
                            # Clean up
                            try:
                                Chem.SanitizeMol(combined)
                            except:
                                pass
                            
                            # Validate the fragmentation
                            if validate_and_clean_fragments(combined):
                                # Determine number of cuts based on fragment count
                                num_cuts = len(frag_set) - 1
                                results.append((f"MMPA_{num_cuts}CUT", combined))
                        except Exception as e:
                            continue
                
                # If RDKit MMPA didn't produce results, fall back
                if not results:
                    return fragment_mol_legacy(mol, maxCuts, "single", include_rings)
                    
            except Exception as e:
                # Fall back to legacy method
                return fragment_mol_legacy(mol, maxCuts, strategy, include_rings)
        else:
            # Use legacy fragmentation
            return fragment_mol_legacy(mol, maxCuts, strategy, include_rings)
        
        return results

    def generate_rgroup_representations(core_smiles, rgroup_smiles):
        """Generate better R-group representations"""
        try:
            # Simple labeling for now
            if '*' in core_smiles and '*' in rgroup_smiles:
                return (core_smiles.replace('*', '[*:1]'), 
                        rgroup_smiles.replace('*', '[*:1]'))
            return core_smiles, rgroup_smiles
        except:
            return core_smiles, rgroup_smiles

    def perform_mmp_analysis(df, min_transform_occurrence, max_cuts=1, 
                           fragmentation_strategy="single", include_rings=False,
                           show_debug=False, fragmentation_method="Auto-select"):
        """Perform MMP analysis with configurable cuts"""
        if df is None or len(df) == 0:
            return None, None
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Map method to strategy
        if fragmentation_method == "RDKit MMPA (Recommended)":
            strategy = "RDKit MMPA"
        elif fragmentation_method == "Legacy Method":
            strategy = fragmentation_strategy
        else:  # Auto-select
            strategy = "Auto-select"
        
        # Step 1: Decompose molecules
        status_text.text(f"Step 1/4: Decomposing molecules ({fragmentation_method})...")
        progress_bar.progress(25)
        
        row_list = []
        fragmentation_stats = {
            "mmpa_single": 0,
            "mmpa_multiple": 0,
            "legacy_single": 0,
            "legacy_multiple": 0,
            "smart_cuts": 0,
            "failed": 0,
            "total": 0
        }
        
        for idx, row in df.iterrows():
            smiles = row['SMILES']
            name = row.get('Name', f"CMPD_{idx}")
            pIC50 = row['pIC50']
            mol = row['mol']
            
            fragmentation_stats["total"] += 1
            
            if mol is None:
                fragmentation_stats["failed"] += 1
                continue
                
            try:
                # Use improved fragmentation
                frag_list = FragmentMol(mol, maxCuts=max_cuts, 
                                      strategy=strategy, 
                                      include_rings=include_rings)
                
                for frag_name, frag_mol in frag_list:
                    # Update stats based on fragmentation type
                    if "MMPA" in frag_name:
                        if "1CUT" in frag_name:
                            fragmentation_stats["mmpa_single"] += 1
                        else:
                            fragmentation_stats["mmpa_multiple"] += 1
                    elif "LEGACY" in frag_name:
                        if "CUTS_" in frag_name:
                            fragmentation_stats["legacy_multiple"] += 1
                        else:
                            fragmentation_stats["legacy_single"] += 1
                    elif "SMART" in frag_name:
                        fragmentation_stats["smart_cuts"] += 1
                    
                    # Validate the fragmentation
                    if validate_and_clean_fragments(frag_mol):
                        # Convert to SMILES
                        try:
                            pair_list = sort_fragments(frag_mol)
                            if len(pair_list) >= 2:
                                # Generate better representations
                                core_smiles = Chem.MolToSmiles(pair_list[0])
                                rgroup_smiles = Chem.MolToSmiles(pair_list[1])
                                
                                tmp_list = [smiles, core_smiles, rgroup_smiles, 
                                          name, pIC50, frag_name]
                                row_list.append(tmp_list)
                        except:
                            fragmentation_stats["failed"] += 1
            except Exception as e:
                fragmentation_stats["failed"] += 1
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
            <p><strong>Method Used:</strong> {fragmentation_method}</p>
            <p><strong>Total molecules processed:</strong> {fragmentation_stats['total']}</p>
            <p><strong>RDKit MMPA single cuts:</strong> {fragmentation_stats['mmpa_single']}</p>
            <p><strong>RDKit MMPA multiple cuts:</strong> {fragmentation_stats['mmpa_multiple']}</p>
            <p><strong>Legacy single cuts:</strong> {fragmentation_stats['legacy_single']}</p>
            <p><strong>Legacy multiple cuts:</strong> {fragmentation_stats['legacy_multiple']}</p>
            <p><strong>Smart cuts:</strong> {fragmentation_stats['smart_cuts']}</p>
            <p><strong>Failed fragmentations:</strong> {fragmentation_stats['failed']}</p>
            <p><strong>Valid fragments generated:</strong> {len(row_list)}</p>
            <p><strong>Unique cores identified:</strong> {row_df['Core'].nunique()}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if show_debug:
            with st.expander("Debug: Row DataFrame", expanded=False):
                st.write(f"Total rows: {len(row_df)}")
                st.dataframe(row_df.head(20))
                st.write("Fragmentation types distribution:")
                st.write(row_df['Fragmentation_Type'].value_counts())
        
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
                    
                    # Generate better R-group representations
                    core_a, rgroup_a = generate_rgroup_representations(
                        reagent_a.Core, reagent_a.R_group
                    )
                    core_b, rgroup_b = generate_rgroup_representations(
                        reagent_b.Core, reagent_b.R_group
                    )
                    
                    # Create transform string
                    transform_str = f"{rgroup_a.replace('*','*-')}>>{rgroup_b.replace('*','*-')}"
                    
                    # Store fragmentation type
                    frag_type_a = reagent_a.Fragmentation_Type
                    frag_type_b = reagent_b.Fragmentation_Type
                    frag_types = f"{frag_type_a}|{frag_type_b}"
                    
                    delta_list.append([
                        reagent_a.SMILES, reagent_a.Core, reagent_a.R_group, reagent_a.Name, 
                        reagent_a.pIC50, frag_type_a,
                        reagent_b.SMILES, reagent_b.Core, reagent_b.R_group, reagent_b.Name, 
                        reagent_b.pIC50, frag_type_b,
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
                # Try with labeled atoms
                rxn = AllChem.ReactionFromSmarts(transform.replace('*-','*'), useSmiles=True)
                rxn_mols.append(rxn)
            except Exception as e:
                try:
                    # Try without labels
                    clean_transform = transform.replace('[*:', '[').replace(']', ']')
                    rxn = AllChem.ReactionFromSmarts(clean_transform.replace('*-','*'), useSmiles=True)
                    rxn_mols.append(rxn)
                except:
                    rxn_mols.append(None)
        
        mmp_df['rxn_mol'] = rxn_mols
        
        # Add fragmentation complexity score
        def get_frag_complexity(frag_types_array):
            """Calculate fragmentation complexity score"""
            complexity = 0
            for types in frag_types_array:
                for t in types.split('|'):
                    if 'MMPA_' in t:
                        # Extract number of cuts from MMPA_2CUT format
                        import re
                        match = re.search(r'MMPA_(\d+)CUT', t)
                        if match:
                            complexity = max(complexity, int(match.group(1)))
                    elif 'CUTS_' in t:
                        # Count number of cuts in LEGACY_CUTS_1_2_3 format
                        cut_count = len(t.split('_')) - 2  # Subtract "LEGACY" and "CUTS"
                        complexity = max(complexity, cut_count)
                    elif t not in ['NO_CUT', 'ERROR']:
                        complexity = max(complexity, 1)
            return complexity
        
        mmp_df['frag_complexity'] = [get_frag_complexity(x) for x in mmp_df['Fragmentation_Types']]
        
        # Add fragmentation method classification
        def get_frag_method(frag_types_array):
            """Determine fragmentation method used"""
            methods = set()
            for types in frag_types_array:
                for t in types.split('|'):
                    if 'MMPA' in t:
                        methods.add('RDKit MMPA')
                    elif 'LEGACY' in t or 'SMART' in t or 'CUT' in t:
                        methods.add('Legacy')
                    elif 'NO_CUT' in t:
                        methods.add('No Cut')
            return ', '.join(sorted(methods)) if methods else 'Unknown'
        
        mmp_df['frag_method'] = [get_frag_method(x) for x in mmp_df['Fragmentation_Types']]
        
        # Step 4: Complete
        status_text.text("Step 4/4: Analysis complete!")
        progress_bar.progress(100)
        status_text.empty()
        progress_bar.empty()
        
        return delta_df, mmp_df

    def plot_stripplot_to_fig(deltas):
        """Create a stripplot figure"""
        fig, ax = plt.subplots(figsize=(4, 1.5))
        if len(deltas) > 0:
            sns.stripplot(x=deltas, ax=ax, jitter=0.2, alpha=0.7, s=5, color='blue')
            ax.axvline(0, ls='--', c='red')
            
            # Set appropriate x limits
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
            <p>• Fragmentation method: <strong>{fragmentation_method}</strong></p>
            <p>• Maximum cuts per molecule: <strong>{max_cuts}</strong></p>
            <p>• Strategy: <strong>{fragmentation_strategy if fragmentation_method == 'Legacy Method' else 'RDKit MMPA'}</strong></p>
            <p>• Allow ring bond cutting: <strong>{'Yes' if include_rings else 'No'}</strong></p>
            <p>• Minimum transform occurrences: <strong>{min_occurrence}</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        delta_df, mmp_df = perform_mmp_analysis(
            df, 
            min_occurrence, 
            max_cuts=max_cuts,
            fragmentation_strategy=fragmentation_strategy,
            fragmentation_method=fragmentation_method,
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
                
                # Calculate fragmentation method distribution
                method_counts = mmp_df['frag_method'].value_counts()
                method_summary = ", ".join([f"{k}: {v}" for k, v in method_counts.items()[:2]])
                if len(method_counts) > 2:
                    method_summary += f", ..."
                col4.metric("Fragmentation Methods", method_summary)
                
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
                                # Show fragmentation method badge
                                method_badge = f'<span style="background-color: #3B82F6; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8rem; margin-right: 5px;">{row["frag_method"]}</span>'
                                complexity_badge = ""
                                if row['frag_complexity'] > 1:
                                    complexity_badge = f'<span style="background-color: #F59E0B; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8rem;">{row["frag_complexity"]} cuts</span>'
                                
                                st.markdown(f"""
                                <div class="transform-card">
                                    <h4>Transform #{i+1} {method_badge} {complexity_badge}</h4>
                                    <p><strong>Transformation:</strong> {row['Transform']}</p>
                                    <p><strong>Mean ΔpIC50:</strong> <span style="color: {'#10B981' if row['mean_delta'] > 0 else '#EF4444'}">{row['mean_delta']:.2f}</span> ± {row['std_delta']:.2f}</p>
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
                                # Show fragmentation method badge
                                method_badge = f'<span style="background-color: #3B82F6; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8rem; margin-right: 5px;">{row["frag_method"]}</span>'
                                complexity_badge = ""
                                if row['frag_complexity'] > 1:
                                    complexity_badge = f'<span style="background-color: #F59E0B; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8rem;">{row["frag_complexity"]} cuts</span>'
                                
                                st.markdown(f"""
                                <div class="transform-card">
                                    <h4>Transform #{i+1} (Negative) {method_badge} {complexity_badge}</h4>
                                    <p><strong>Transformation:</strong> {row['Transform']}</p>
                                    <p><strong>Mean ΔpIC50:</strong> <span style="color: {'#10B981' if row['mean_delta'] > 0 else '#EF4444'}">{row['mean_delta']:.2f}</span> ± {row['std_delta']:.2f}</p>
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
                                        st.dataframe(examples_df)
                
                # Show all transforms table with filtering options
                if len(mmp_df_sorted) > 0:
                    st.markdown('<h3 class="section-header">📋 All Transformations</h3>', unsafe_allow_html=True)
                    
                    # Add filtering options
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        filter_method = st.selectbox(
                            "Filter by fragmentation method",
                            options=["All", "RDKit MMPA", "Legacy", "Mixed"],
                            index=0
                        )
                    
                    with col2:
                        filter_direction = st.selectbox(
                            "Filter by effect direction",
                            options=["All", "Positive only (Δ>0)", "Negative only (Δ<0)", "Neutral (Δ≈0)"],
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
                    
                    if filter_method == "RDKit MMPA":
                        filtered_df = filtered_df[filtered_df['frag_method'].str.contains('RDKit MMPA')]
                    elif filter_method == "Legacy":
                        filtered_df = filtered_df[filtered_df['frag_method'].str.contains('Legacy')]
                    elif filter_method == "Mixed":
                        filtered_df = filtered_df[filtered_df['frag_method'].str.contains(',')]
                    
                    if filter_direction == "Positive only (Δ>0)":
                        filtered_df = filtered_df[filtered_df['mean_delta'] > 0]
                    elif filter_direction == "Negative only (Δ<0)":
                        filtered_df = filtered_df[filtered_df['mean_delta'] < 0]
                    elif filter_direction == "Neutral (Δ≈0)":
                        filtered_df = filtered_df[abs(filtered_df['mean_delta']) < 0.5]
                    
                    filtered_df = filtered_df[filtered_df['Count'] >= min_frequency]
                    
                    st.info(f"Showing {len(filtered_df)} transforms after filtering")
                    
                    if show_all_transforms:
                        display_df = filtered_df
                    else:
                        display_df = filtered_df.head(transforms_to_display)
                    
                    # Enhanced table display
                    display_columns = ['Transform', 'Count', 'mean_delta', 'std_delta', 'frag_complexity', 'frag_method']
                    st.dataframe(display_df[display_columns].rename(
                        columns={
                            'mean_delta': 'Mean ΔpIC50',
                            'std_delta': 'Std ΔpIC50',
                            'frag_complexity': 'Cuts',
                            'frag_method': 'Method'
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
    
    ### New Feature: Advanced Fragmentation Methods
    
    You can now choose between different fragmentation algorithms:
    
    - **RDKit MMPA (Recommended)**: Uses RDKit's optimized fragmentation algorithm (fastest and most reliable)
    - **Legacy Method**: Original bond-breaking method with full control over cutting strategy
    - **Auto-select**: Automatically chooses the best method based on your parameters
    
    ### How to use:
    1. **Upload your data** using the sidebar on the left
    2. **Choose fragmentation method** (RDKit MMPA is recommended for most cases)
    3. **Configure fragmentation parameters** (number of cuts, strategy for Legacy method)
    4. **Set analysis parameters** like minimum transform occurrences
    5. **View results** including top positive/negative transformations
    6. **Export findings** for further analysis
    
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
    - **Configurable fragmentation methods** for different analysis needs
    - **Advanced filtering** by fragmentation method and effect direction
    
    ### Troubleshooting:
    If you encounter errors:
    1. **NumPy compatibility**: Install `numpy<2` with `pip install "numpy<2"`
    2. **Invalid SMILES**: Check your SMILES strings are valid
    3. **Memory issues with many cuts**: Reduce max cuts or use RDKit MMPA method

    ### References:
    - Hussain, J. & Rea, C. (2010). Computationally efficient algorithm to identify matched molecular pairs (MMPs) in large data sets. *Journal of Chemical Information and Modeling*, 50(3), 339-348. https://doi.org/10.1021/ci900450m
    - Dossetter, A. G., Griffen, E. J., & Leach, A. G. (2013). Matched molecular pair analysis in drug discovery. *Drug Discovery Today*, 18(15-16), 724-731. https://doi.org/10.1016/j.drudis.2013.03.003
    - Wassermann, A. M., Dimova, D., Iyer, P., & Bajorath, J., Advances in computational medicinal chemistry: matched molecular pair analysis. Drug Development Research, 73 (2012): 518-527. https://doi.org/10.1002/ddr.21045
    - Tyrchan, Christian, and Emma Evertsson. "Matched molecular pair analysis in short: algorithms, applications and limitations," Computational and Structural Biotechnology Journal 15 (2017): 86-90 https://doi.org/10.1016/j.csbj.2016.12.003
    
    
    ⬅️ **Upload a CSV file in the sidebar to get started!**
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; font-size: 0.9rem;">
    <p>MMP Analysis Tool v2.0 | Advanced Fragmentation Edition | Built with Streamlit, RDKit, and Pandas</p>
    <p>For research use only. Always validate computational predictions with experimental data.</p>
</div>
""", unsafe_allow_html=True)
