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
from tqdm.auto import tqdm

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
.cut-level {
    background-color: #E0F2FE;
    border-radius: 5px;
    padding: 0.5rem;
    margin: 0.5rem 0;
    border-left: 3px solid #0EA5E9;
}
.success-box {
    background-color: #D1FAE5;
    border-left: 4px solid #10B981;
    padding: 1rem;
    border-radius: 5px;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🧪 Advanced MMP Analysis Tool</h1>', unsafe_allow_html=True)

# Try to import RDKit with error handling
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw
    from rdkit.Chem.rdMMPA import FragmentMol
    from rdkit.Chem.rdRGroupDecomposition import RGroupDecompose
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.Chem.rdmolops import ReplaceCore, ReplaceSubstructs
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

# Helper functions for RDKit utilities
class UsefulRDKitUtils:
    @staticmethod
    def get_largest_fragment(mol):
        """Get the largest fragment from a molecule"""
        if mol is None:
            return None
        frags = Chem.GetMolFrags(mol, asMols=True)
        if not frags:
            return mol
        return max(frags, key=lambda x: x.GetNumAtoms())
    
    @staticmethod
    def cleanup_fragment(mol):
        """Replace atom map numbers with Hydrogens and remove all Hs"""
        if mol is None:
            return None, 0
        
        rgroup_count = 0
        for atm in mol.GetAtoms():
            atm.SetAtomMapNum(0)
            if atm.GetAtomicNum() == 0:
                rgroup_count += 1
                atm.SetAtomicNum(1)
        
        # Create a copy and remove hydrogens
        mol_copy = Chem.RemoveAllHs(mol)
        return mol_copy, rgroup_count
    
    @staticmethod
    def generate_fragments(mol, min_fraction=0.67):
        """
        Generate fragments using RDKit's FragmentMol
        Similar to the reference code but more configurable
        """
        if mol is None:
            return pd.DataFrame(columns=["Scaffold", "NumAtoms", "NumRgroups"])
        
        # Generate molecule fragments
        frag_list = FragmentMol(mol)
        
        # Flatten the output into a single list
        flat_frag_list = [x for x in itertools.chain(*frag_list) if x]
        
        # Get largest fragment from each
        flat_frag_list = [UsefulRDKitUtils.get_largest_fragment(x) for x in flat_frag_list]
        
        # Filter by size
        num_mol_atoms = mol.GetNumAtoms()
        flat_frag_list = [x for x in flat_frag_list 
                         if x.GetNumAtoms() / num_mol_atoms >= min_fraction]
        
        # Clean up fragments
        flat_frag_list = [UsefulRDKitUtils.cleanup_fragment(x) for x in flat_frag_list]
        
        # Convert to SMILES
        frag_smiles_list = []
        for frag_mol, rgroup_count in flat_frag_list:
            if frag_mol:
                frag_smiles_list.append([
                    Chem.MolToSmiles(frag_mol),
                    frag_mol.GetNumAtoms(),
                    rgroup_count
                ])
        
        # Add the original molecule as a scaffold
        frag_smiles_list.append([
            Chem.MolToSmiles(mol),
            mol.GetNumAtoms(),
            1
        ])
        
        # Create DataFrame
        frag_df = pd.DataFrame(frag_smiles_list, 
                              columns=["Scaffold", "NumAtoms", "NumRgroups"])
        
        # Remove duplicates
        frag_df = frag_df.drop_duplicates("Scaffold")
        
        return frag_df
    
    @staticmethod
    def find_scaffolds(df_in, min_fraction=0.67):
        """
        Generate scaffolds for a set of molecules
        Returns: molecule-scaffold mapping and unique scaffolds
        """
        df_list = []
        
        # Use tqdm for progress if available
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(df_in[["SMILES", "Name", "mol"]].values, 
                           desc="Generating scaffolds")
        except:
            iterator = df_in[["SMILES", "Name", "mol"]].values
        
        for smiles, name, mol in iterator:
            if mol is not None:
                tmp_df = UsefulRDKitUtils.generate_fragments(mol, min_fraction)
                tmp_df['Name'] = name
                tmp_df['SMILES'] = smiles
                df_list.append(tmp_df)
        
        if not df_list:
            return pd.DataFrame(), pd.DataFrame()
        
        # Combine all dataframes
        mol_df = pd.concat(df_list, ignore_index=True)
        
        # Collect unique scaffolds
        scaffold_list = []
        for scaffold, group in mol_df.groupby("Scaffold"):
            scaffold_list.append([
                scaffold,
                len(group.Name.unique()),
                group.NumAtoms.iloc[0],
                group.NumRgroups.mean()
            ])
        
        scaffold_df = pd.DataFrame(scaffold_list, 
                                  columns=["Scaffold", "Count", "NumAtoms", "AvgRgroups"])
        
        # Filter scaffolds that occur in reasonable number of molecules
        num_df_rows = len(df_in)
        scaffold_df = scaffold_df.query("Count > 1")  # At least 2 molecules
        
        # Sort by frequency
        scaffold_df = scaffold_df.sort_values(["Count", "NumAtoms"], 
                                             ascending=[False, False])
        
        return mol_df, scaffold_df

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
        
        min_scaffold_occurrence = st.slider(
            "Minimum scaffold occurrences",
            2, 50, 3,
            help="Minimum number of molecules sharing a scaffold"
        )
        
        # Fragment generation parameters
        st.markdown("### ✂️ Fragment Generation")
        
        min_fraction = st.slider(
            "Minimum fragment size fraction", 
            0.1, 0.9, 0.67, 0.01,
            help="Minimum fraction of atoms in fragment relative to original molecule"
        )
        
        max_rgroups = st.slider(
            "Maximum R-groups per scaffold",
            1, 10, 4,
            help="Maximum number of R-groups to consider per scaffold"
        )
        
        # Advanced options
        with st.expander("Advanced Options"):
            sanitize_molecules = st.checkbox("Sanitize molecules", value=True)
            kekulize_molecules = st.checkbox("Kekulize molecules", value=False)
            remove_salts = st.checkbox("Remove salts", value=True)
            normalize_molecules = st.checkbox("Normalize molecules", value=True)
            
        # Display options
        st.markdown("### 👀 Display Options")
        show_all_transforms = st.checkbox("Show all transformations", value=False)
        transforms_to_display = st.slider(
            "Number of transforms to display", 
            1, 100, 20, 
            disabled=show_all_transforms
        )
        
        # Export options
        st.markdown("### 💾 Export")
        save_results = st.checkbox("Save results to Excel", value=False)

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
    # Load and process data
    @st.cache_data
    def load_and_process_data(file, sanitize=True, kekulize=False, remove_salts=True, normalize=True):
        """Load and preprocess chemical data"""
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
        
        for idx, row in df.iterrows():
            try:
                smiles = str(row['SMILES'])
                
                # Parse molecule
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    errors.append(f"Row {idx}: Invalid SMILES '{smiles}'")
                    molecules.append(None)
                    continue
                
                # Sanitize
                if sanitize:
                    try:
                        Chem.SanitizeMol(mol)
                    except:
                        pass
                
                # Kekulize
                if kekulize:
                    try:
                        Chem.Kekulize(mol, clearAromaticFlags=True)
                    except:
                        pass
                
                # Remove salts (get largest fragment)
                if remove_salts:
                    frags = Chem.GetMolFrags(mol, asMols=True)
                    if frags:
                        mol = max(frags, key=lambda x: x.GetNumAtoms())
                
                # Normalize
                if normalize:
                    try:
                        mol = Chem.RemoveHs(mol)
                        # Add more normalization steps as needed
                    except:
                        pass
                
                molecules.append(mol)
                
            except Exception as e:
                errors.append(f"Row {idx}: Error processing '{smiles}' - {str(e)}")
                molecules.append(None)
        
        df['mol'] = molecules
        
        # Show errors
        if errors:
            with st.expander("⚠️ Processing Errors", expanded=False):
                for error in errors[:10]:
                    st.warning(error)
                if len(errors) > 10:
                    st.info(f"... and {len(errors)-10} more errors")
        
        # Remove invalid molecules
        valid_df = df[df['mol'].notna()].copy()
        if len(valid_df) < len(df):
            st.warning(f"Removed {len(df) - len(valid_df)} rows with invalid molecules")
        
        return valid_df
    
    # Get parameters
    sanitize = sanitize_molecules if 'sanitize_molecules' in locals() else True
    kekulize = kekulize_molecules if 'kekulize_molecules' in locals() else False
    remove_salts_val = remove_salts if 'remove_salts' in locals() else True
    normalize_val = normalize_molecules if 'normalize_molecules' in locals() else True
    
    # Load data
    df = load_and_process_data(
        uploaded_file, 
        sanitize=sanitize, 
        kekulize=kekulize,
        remove_salts=remove_salts_val,
        normalize=normalize_val
    )
    
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
                            name = row['Name']
                            st.image(img, caption=f"{name} (pIC50: {row['pIC50']:.2f})")
                    except:
                        pass
        
        # Perform scaffold analysis
        st.markdown('<h2 class="section-header">🔬 Scaffold Analysis</h2>', unsafe_allow_html=True)
        
        with st.spinner("Generating scaffolds..."):
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Step 1/4: Finding common scaffolds...")
            progress_bar.progress(25)
            
            # Find scaffolds using the improved method
            mol_df, scaffold_df = UsefulRDKitUtils.find_scaffolds(df, min_fraction)
            
            if len(scaffold_df) == 0:
                st.error("No common scaffolds found. Try adjusting the minimum fraction parameter.")
                st.stop()
            
            status_text.text("Step 2/4: Analyzing scaffold distribution...")
            progress_bar.progress(50)
            
            # Filter by minimum occurrence
            scaffold_df_filtered = scaffold_df.query(f"Count >= {min_scaffold_occurrence}")
            
            if len(scaffold_df_filtered) == 0:
                st.warning(f"No scaffolds found with {min_scaffold_occurrence}+ occurrences. Try reducing the threshold.")
                scaffold_df_filtered = scaffold_df.head(10)
            
            status_text.text("Step 3/4: Performing R-group decomposition...")
            progress_bar.progress(75)
            
            # Function to perform MMP analysis
            def perform_mmp_analysis_with_scaffolds(scaffold_df_filtered, mol_df, activity_df, min_occurrence):
                """Perform MMP analysis using scaffolds"""
                all_mmp_pairs = []
                transform_stats = {}
                
                # Process each scaffold
                for _, scaffold_row in scaffold_df_filtered.iterrows():
                    scaffold_smiles = scaffold_row['Scaffold']
                    
                    # Get molecules with this scaffold
                    match_df = mol_df.query("Scaffold == @scaffold_smiles")
                    merge_df = match_df.merge(activity_df, on=["SMILES", "Name"])
                    
                    if len(merge_df) < 2:
                        continue
                    
                    # Perform R-group decomposition
                    scaffold_mol = Chem.MolFromSmiles(scaffold_smiles)
                    if scaffold_mol is None:
                        continue
                    
                    try:
                        # Get molecules as RDKit objects
                        mol_list = []
                        valid_indices = []
                        for idx, row in merge_df.iterrows():
                            if row['mol'] is not None:
                                mol_list.append(row['mol'])
                                valid_indices.append(idx)
                        
                        if len(mol_list) < 2:
                            continue
                        
                        # Perform R-group decomposition
                        rgroup_results, rgroup_miss = RGroupDecompose(
                            [scaffold_mol],
                            mol_list,
                            asSmiles=True,
                            asRows=False
                        )
                        
                        if rgroup_results and len(rgroup_results) > 0:
                            # Convert to DataFrame
                            rgroup_df = pd.DataFrame(rgroup_results)
                            
                            # Add activity data
                            activity_cols = merge_df.iloc[valid_indices][['SMILES', 'Name', 'pIC50']].reset_index(drop=True)
                            rgroup_df = pd.concat([rgroup_df, activity_cols], axis=1)
                            
                            # Process each R-group position
                            rgroup_columns = [col for col in rgroup_df.columns if col.startswith('R')]
                            
                            for rgroup_col in rgroup_columns:
                                # Group by R-group and find pairs
                                for rgroup_smiles, group in rgroup_df.groupby(rgroup_col):
                                    if len(group) >= 2:
                                        # Generate all unique pairs
                                        for i in range(len(group)):
                                            for j in range(i+1, len(group)):
                                                mol1 = group.iloc[i]
                                                mol2 = group.iloc[j]
                                                
                                                if mol1['SMILES'] == mol2['SMILES']:
                                                    continue
                                                
                                                # Create transform string
                                                transform = f"{mol1[rgroup_col]}>>{mol2[rgroup_col]}"
                                                
                                                # Calculate delta pIC50
                                                delta = mol2['pIC50'] - mol1['pIC50']
                                                
                                                # Store pair
                                                all_mmp_pairs.append({
                                                    'Scaffold': scaffold_smiles,
                                                    'RGroup_Position': rgroup_col,
                                                    'Transform': transform,
                                                    'SMILES_1': mol1['SMILES'],
                                                    'Name_1': mol1['Name'],
                                                    'pIC50_1': mol1['pIC50'],
                                                    'RGroup_1': mol1[rgroup_col],
                                                    'SMILES_2': mol2['SMILES'],
                                                    'Name_2': mol2['Name'],
                                                    'pIC50_2': mol2['pIC50'],
                                                    'RGroup_2': mol2[rgroup_col],
                                                    'Delta': delta,
                                                    'Num_RGroups': len(rgroup_columns)
                                                })
                                                
                                                # Update transform statistics
                                                if transform not in transform_stats:
                                                    transform_stats[transform] = {
                                                        'count': 0,
                                                        'deltas': [],
                                                        'scaffolds': set(),
                                                        'rgroups': set([rgroup_col])
                                                    }
                                                transform_stats[transform]['count'] += 1
                                                transform_stats[transform]['deltas'].append(delta)
                                                transform_stats[transform]['scaffolds'].add(scaffold_smiles)
                                                transform_stats[transform]['rgroups'].add(rgroup_col)
                                                
                    except Exception as e:
                        st.warning(f"Error processing scaffold {scaffold_smiles}: {str(e)}")
                        continue
                
                # Convert to DataFrames
                pairs_df = pd.DataFrame(all_mmp_pairs) if all_mmp_pairs else pd.DataFrame()
                
                # Create transforms DataFrame
                transform_list = []
                for transform, stats in transform_stats.items():
                    if stats['count'] >= min_occurrence:
                        transform_list.append({
                            'Transform': transform,
                            'Count': stats['count'],
                            'Mean_Delta': np.mean(stats['deltas']),
                            'Std_Delta': np.std(stats['deltas']),
                            'Min_Delta': min(stats['deltas']),
                            'Max_Delta': max(stats['deltas']),
                            'Num_Scaffolds': len(stats['scaffolds']),
                            'Num_RGroups': len(stats['rgroups']),
                            'Deltas': stats['deltas'],
                            'Scaffolds': list(stats['scaffolds'])[:5]  # Limit to first 5
                        })
                
                transforms_df = pd.DataFrame(transform_list) if transform_list else pd.DataFrame()
                
                # Sort transforms by absolute mean delta
                if len(transforms_df) > 0:
                    transforms_df['Abs_Mean_Delta'] = transforms_df['Mean_Delta'].abs()
                    transforms_df = transforms_df.sort_values('Abs_Mean_Delta', ascending=False)
                
                return pairs_df, transforms_df
            
            # Perform MMP analysis
            pairs_df, transforms_df = perform_mmp_analysis_with_scaffolds(
                scaffold_df_filtered, mol_df, df, min_occurrence
            )
            
            status_text.text("Step 4/4: Analysis complete!")
            progress_bar.progress(100)
            status_text.empty()
            progress_bar.empty()
        
        # Display results
        st.success(f"Analysis complete! Found {len(pairs_df)} molecular pairs and {len(transforms_df)} significant transforms.")
        
        # Show scaffold statistics
        st.markdown('<h3 class="section-header">🏗️ Top Scaffolds</h3>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Scaffolds", len(scaffold_df))
        with col2:
            st.metric("Filtered Scaffolds", len(scaffold_df_filtered))
        with col3:
            st.metric("Max Occurrences", scaffold_df_filtered['Count'].max() if len(scaffold_df_filtered) > 0 else 0)
        with col4:
            st.metric("Avg R-groups", f"{scaffold_df_filtered['AvgRgroups'].mean():.1f}" if len(scaffold_df_filtered) > 0 else 0)
        
        # Display top scaffolds
        st.dataframe(
            scaffold_df_filtered.head(10)[['Scaffold', 'Count', 'NumAtoms', 'AvgRgroups']]
            .rename(columns={
                'Scaffold': 'Scaffold SMILES',
                'Count': 'Molecules',
                'NumAtoms': 'Atoms',
                'AvgRgroups': 'Avg R-groups'
            })
        )
        
        # Show scaffold visualization
        if len(scaffold_df_filtered) > 0:
            st.markdown('<h4>Top Scaffold Visualization</h4>', unsafe_allow_html=True)
            top_scaffold = scaffold_df_filtered.iloc[0]['Scaffold']
            scaffold_mol = Chem.MolFromSmiles(top_scaffold)
            
            if scaffold_mol:
                col1, col2 = st.columns([1, 2])
                with col1:
                    img = Draw.MolToImage(scaffold_mol, size=(300, 300))
                    st.image(img, caption=f"Top Scaffold ({scaffold_df_filtered.iloc[0]['Count']} molecules)")
                with col2:
                    st.markdown(f"""
                    **Scaffold Information:**
                    - **SMILES:** `{top_scaffold}`
                    - **Occurrences:** {scaffold_df_filtered.iloc[0]['Count']} molecules
                    - **Number of Atoms:** {scaffold_df_filtered.iloc[0]['NumAtoms']}
                    - **Average R-groups:** {scaffold_df_filtered.iloc[0]['AvgRgroups']:.1f}
                    """)
        
        # Show MMP analysis results
        if len(transforms_df) > 0:
            st.markdown('<h3 class="section-header">🔄 Top Transformations</h3>', unsafe_allow_html=True)
            
            # Display transform statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Transforms", len(transforms_df))
            with col2:
                positive = len(transforms_df[transforms_df['Mean_Delta'] > 0])
                st.metric("Positive ΔpIC50", positive)
            with col3:
                negative = len(transforms_df[transforms_df['Mean_Delta'] < 0])
                st.metric("Negative ΔpIC50", negative)
            
            # Function to create reaction image
            def get_transform_image(transform_smarts):
                """Create reaction image from transform SMARTS"""
                try:
                    parts = transform_smarts.split('>>')
                    if len(parts) == 2:
                        # Add attachment points
                        left = Chem.MolFromSmiles(parts[0] + '[*]')
                        right = Chem.MolFromSmiles(parts[1] + '[*]')
                        
                        if left and right:
                            # Create reaction
                            rxn = AllChem.ChemicalReaction()
                            rxn.AddReactantTemplate(left)
                            rxn.AddProductTemplate(right)
                            
                            # Draw reaction
                            img = Draw.ReactionToImage(rxn, subImgSize=(200, 150))
                            buffered = io.BytesIO()
                            img.save(buffered, format="PNG")
                            return base64.b64encode(buffered.getvalue()).decode()
                except:
                    pass
                return None
            
            # Show top positive transforms
            st.markdown('<h4>Top Positive Transformations</h4>', unsafe_allow_html=True)
            positive_transforms = transforms_df[transforms_df['Mean_Delta'] > 0].head(5)
            
            for idx, (_, transform_row) in enumerate(positive_transforms.iterrows()):
                with st.container():
                    col1, col2, col3 = st.columns([1, 2, 1])
                    
                    with col1:
                        img_base64 = get_transform_image(transform_row['Transform'])
                        if img_base64:
                            st.markdown(f'<img src="data:image/png;base64,{img_base64}" width="250">', 
                                      unsafe_allow_html=True)
                        else:
                            st.info("Reaction image not available")
                    
                    with col2:
                        st.markdown(f"""
                        <div class="transform-card">
                        <h4>Transform #{idx+1}</h4>
                        <p><strong>Transformation:</strong> {transform_row['Transform']}</p>
                        <p><strong>Mean ΔpIC50:</strong> {transform_row['Mean_Delta']:.2f} ± {transform_row['Std_Delta']:.2f}</p>
                        <p><strong>Occurrences:</strong> {transform_row['Count']}</p>
                        <p><strong>Range:</strong> {transform_row['Min_Delta']:.2f} to {transform_row['Max_Delta']:.2f}</p>
                        <p><strong>Scaffolds:</strong> {transform_row['Num_Scaffolds']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        # Create strip plot
                        fig, ax = plt.subplots(figsize=(3, 1.5))
                        deltas = transform_row['Deltas']
                        if len(deltas) > 0:
                            sns.stripplot(x=deltas, ax=ax, jitter=0.3, alpha=0.7, s=8, color='green')
                            ax.axvline(0, ls='--', c='red', alpha=0.5)
                            ax.set_xlabel('ΔpIC50')
                            ax.set_yticks([])
                            ax.set_title('Distribution')
                        plt.tight_layout()
                        st.pyplot(fig)
            
            # Show top negative transforms
            st.markdown('<h4>Top Negative Transformations</h4>', unsafe_allow_html=True)
            negative_transforms = transforms_df[transforms_df['Mean_Delta'] < 0].head(5)
            
            for idx, (_, transform_row) in enumerate(negative_transforms.iterrows()):
                with st.container():
                    col1, col2, col3 = st.columns([1, 2, 1])
                    
                    with col1:
                        img_base64 = get_transform_image(transform_row['Transform'])
                        if img_base64:
                            st.markdown(f'<img src="data:image/png;base64,{img_base64}" width="250">', 
                                      unsafe_allow_html=True)
                        else:
                            st.info("Reaction image not available")
                    
                    with col2:
                        st.markdown(f"""
                        <div class="transform-card">
                        <h4>Transform #{idx+1}</h4>
                        <p><strong>Transformation:</strong> {transform_row['Transform']}</p>
                        <p><strong>Mean ΔpIC50:</strong> {transform_row['Mean_Delta']:.2f} ± {transform_row['Std_Delta']:.2f}</p>
                        <p><strong>Occurrences:</strong> {transform_row['Count']}</p>
                        <p><strong>Range:</strong> {transform_row['Min_Delta']:.2f} to {transform_row['Max_Delta']:.2f}</p>
                        <p><strong>Scaffolds:</strong> {transform_row['Num_Scaffolds']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        # Create strip plot
                        fig, ax = plt.subplots(figsize=(3, 1.5))
                        deltas = transform_row['Deltas']
                        if len(deltas) > 0:
                            sns.stripplot(x=deltas, ax=ax, jitter=0.3, alpha=0.7, s=8, color='red')
                            ax.axvline(0, ls='--', c='red', alpha=0.5)
                            ax.set_xlabel('ΔpIC50')
                            ax.set_yticks([])
                            ax.set_title('Distribution')
                        plt.tight_layout()
                        st.pyplot(fig)
            
            # Show all transforms table
            st.markdown('<h3 class="section-header">📋 All Transformations</h3>', unsafe_allow_html=True)
            
            if show_all_transforms:
                display_df = transforms_df
            else:
                display_df = transforms_df.head(transforms_to_display)
            
            # Display table
            display_columns = ['Transform', 'Count', 'Mean_Delta', 'Std_Delta', 
                             'Min_Delta', 'Max_Delta', 'Num_Scaffolds', 'Num_RGroups']
            
            st.dataframe(
                display_df[display_columns]
                .rename(columns={
                    'Mean_Delta': 'Mean Δ',
                    'Std_Delta': 'Std Δ',
                    'Min_Delta': 'Min Δ',
                    'Max_Delta': 'Max Δ',
                    'Num_Scaffolds': 'Scaffolds',
                    'Num_RGroups': 'R-Groups'
                })
                .round(3)
            )
            
            # Export results
            if save_results:
                st.markdown('<h3 class="section-header">💾 Export Results</h3>', unsafe_allow_html=True)
                
                @st.cache_data
                def convert_df_to_excel(df_dict):
                    """Convert multiple DataFrames to Excel"""
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        for sheet_name, df in df_dict.items():
                            if len(df) > 0:
                                df.to_excel(writer, index=False, sheet_name=sheet_name)
                    return output.getvalue()
                
                # Prepare data for export
                export_data = {
                    'Scaffolds': scaffold_df_filtered,
                    'Transformations': transforms_df,
                    'Molecular_Pairs': pairs_df,
                    'Original_Data': df[['SMILES', 'Name', 'pIC50']]
                }
                
                excel_data = convert_df_to_excel(export_data)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📥 Download as CSV",
                        data=transforms_df.to_csv(index=False).encode('utf-8'),
                        file_name="mmp_transforms.csv",
                        mime="text/csv"
                    )
                with col2:
                    st.download_button(
                        label="📥 Download Complete Analysis (Excel)",
                        data=excel_data,
                        file_name="advanced_mmp_analysis.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        else:
            st.warning(f"No significant transformations found with {min_occurrence}+ occurrences.")
            
    else:
        st.error("No valid data loaded. Please check your input file.")
else:
    # Show welcome message
    st.markdown("""
    ## Welcome to the Advanced MMP Analysis Tool! 🧪
    
    This tool performs **Matched Molecular Pair (MMP) analysis using advanced RDKit fragmentation methods**.
    
    ### Key Features:
    
    1. **Advanced Scaffold Detection**: Uses RDKit's FragmentMol for intelligent fragmentation
    2. **R-Group Decomposition**: Automatically identifies and labels R-group positions
    3. **Statistical Analysis**: Calculates mean ΔpIC50 with standard deviations
    4. **Multi-Scaffold Support**: Identifies transformations across multiple scaffolds
    5. **Interactive Visualization**: View reaction diagrams and distribution plots
    
    ### How to use:
    
    1. **Upload your data** using the sidebar (CSV with SMILES, pIC50 columns)
    2. **Configure analysis parameters**:
       - Minimum transform occurrences
       - Minimum scaffold occurrences  
       - Fragment size fraction
    3. **View results**:
       - Top scaffolds
       - Significant transformations
       - Statistical distributions
    4. **Export findings** for further analysis
    
    ### Expected CSV format:
    ```
    SMILES,pIC50,Name (optional)
    CCCCN,6.5,Compound1
    CCCCCO,7.2,Compound2
    ...
    ```
    
    ⬅️ **Upload a CSV file in the sidebar to get started!**
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; font-size: 0.9rem;">
<p>Advanced MMP Analysis Tool v3.0 | Built with Streamlit and RDKit</p>
<p>For research use only. Always validate computational predictions with experimental data.</p>
</div>
""", unsafe_allow_html=True)

