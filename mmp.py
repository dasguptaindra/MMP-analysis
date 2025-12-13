# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import warnings
from collections import defaultdict
from itertools import combinations
from operator import itemgetter
from io import BytesIO

# Suppress warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Advanced MMP Analysis Tool (No Filters)",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1E3A8A; text-align: center; margin-bottom: 2rem; }
    .section-header { font-size: 1.8rem; color: #2563EB; margin-top: 2rem; margin-bottom: 1rem; border-bottom: 2px solid #E5E7EB; padding-bottom: 0.5rem; }
    .mmp-image { max-width: 100%; border: 1px solid #ddd; border-radius: 5px; padding: 5px; }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🧪 Advanced MMP Analysis Tool (Unfiltered)</h1>', unsafe_allow_html=True)

# Try to import RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Chem import Descriptors
    from rdkit.Chem.rdMMPA import FragmentMol
    RDKIT_AVAILABLE = True
except ImportError:
    st.error("RDKit not available. Please install with: pip install rdkit")
    RDKIT_AVAILABLE = False

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

if RDKIT_AVAILABLE:
    def remove_map_nums(mol):
        """Remove atom map numbers from a molecule"""
        for atm in mol.GetAtoms():
            atm.SetAtomMapNum(0)
        return mol
    
    def sort_fragments(mol):
        """Transform a molecule with multiple fragments into a list of molecules sorted by size"""
        frag_list = list(Chem.GetMolFrags(mol, asMols=True))
        [remove_map_nums(x) for x in frag_list]
        frag_num_atoms_list = [(x.GetNumAtoms(), x) for x in frag_list]
        frag_num_atoms_list.sort(key=itemgetter(0), reverse=True)
        return [x[1] for x in frag_num_atoms_list]
    
    def generate_fragments_exhaustive(mol):
        """Generate fragments using RDKit's FragmentMol with exhaustive single cuts"""
        # maxCuts=1 ensures we get simple Core + R-group pairs
        frag_list = FragmentMol(mol, maxCuts=1)
        results = []
        for _, frag_mol in frag_list:
            pair_list = sort_fragments(frag_mol)
            if len(pair_list) == 2:
                core_mol, rgroup_mol = pair_list[0], pair_list[1]
                core_smiles = Chem.MolToSmiles(core_mol)
                rgroup_smiles = Chem.MolToSmiles(rgroup_mol)
                
                # Check if attachment point is present
                if '*' in core_smiles and '*' in rgroup_smiles:
                    results.append({
                        'core_mol': core_mol,
                        'rgroup_mol': rgroup_mol,
                        'core_smiles': core_smiles,
                        'rgroup_smiles': rgroup_smiles,
                        'core_size': core_mol.GetNumAtoms(),
                        'rgroup_size': rgroup_mol.GetNumAtoms()
                    })
        return results

    @st.cache_data
    def load_and_preprocess_data(file):
        """
        Load and preprocess CSV data WITHOUT property filtering.
        """
        if file is None:
            return None
        
        try:
            df = pd.read_csv(file)
            
            # Normalize column names
            cols = {c.lower(): c for c in df.columns}
            
            if 'smiles' not in cols:
                st.error("CSV must contain 'SMILES' column")
                return None
            
            # Rename critical columns
            df.rename(columns={cols['smiles']: 'SMILES'}, inplace=True)
            
            if 'pic50' in cols:
                df.rename(columns={cols['pic50']: 'pIC50'}, inplace=True)
            elif 'pIC50' not in df.columns:
                st.warning("pIC50 column not found. Using random values for demonstration.")
                np.random.seed(42)
                df['pIC50'] = np.random.uniform(4.0, 8.0, len(df))
            
            if 'Name' not in df.columns:
                if 'name' in cols:
                    df.rename(columns={cols['name']: 'Name'}, inplace=True)
                else:
                    df['Name'] = [f"Compound_{i+1}" for i in range(len(df))]
            
            # Convert SMILES to molecules
            molecules = []
            valid_indices = []
            
            for idx, row in df.iterrows():
                try:
                    smi = str(row['SMILES']).strip()
                    mol = Chem.MolFromSmiles(smi)
                    if mol is not None:
                        # --- MODIFICATION: NO MW FILTERING HERE ---
                        molecules.append(mol)
                        valid_indices.append(idx)
                    else:
                        st.warning(f"Invalid SMILES at row {idx}: {smi}")
                except Exception as e:
                    st.warning(f"Error processing row {idx}: {e}")
            
            if not molecules:
                st.error("No valid molecules found")
                return None
            
            # Create final dataframe
            final_df = df.iloc[valid_indices].copy()
            final_df['mol'] = molecules
            
            # Calculate properties for display only
            final_df['MW'] = [Descriptors.MolWt(mol) for mol in molecules]
            
            st.success(f"Loaded {len(final_df)} valid compounds (No filters applied).")
            return final_df
            
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None

    def perform_mmp_analysis_single_cut(df, min_pairs_per_core, min_transform_occurrence):
        """Perform MMP analysis with single cuts only"""
        
        status_text = st.empty()
        progress_bar = st.progress(0)
        
        # Step 1: Generate fragments
        status_text.text("Step 1/4: Generating fragments (Exhaustive Single Cut)...")
        progress_bar.progress(10)
        
        compound_fragments = {}
        
        for idx, row in df.iterrows():
            mol = row['mol']
            # Generate ALL single cuts
            fragments = generate_fragments_exhaustive(mol)
            
            if fragments:
                compound_fragments[idx] = {
                    'name': row['Name'],
                    'smiles': row['SMILES'],
                    'pIC50': row['pIC50'],
                    'fragments': fragments
                }
            
        if not compound_fragments:
            st.error("No fragments generated.")
            return None, None
        
        # Step 2: Group by core
        status_text.text("Step 2/4: Grouping by common cores...")
        progress_bar.progress(40)
        
        core_to_compounds = defaultdict(list)
        
        for comp_idx, comp_data in compound_fragments.items():
            seen_cores = set()
            for frag in comp_data['fragments']:
                core_smiles = frag['core_smiles']
                
                # Standardize core string for grouping
                core_key = core_smiles.replace('[*]', '*').strip()
                
                if core_key not in seen_cores:
                    core_to_compounds[core_key].append({
                        'comp_idx': comp_idx,
                        'name': comp_data['name'],
                        'smiles': comp_data['smiles'],
                        'pIC50': comp_data['pIC50'],
                        'rgroup_smiles': frag['rgroup_smiles'],
                        'core_size': frag['core_size']
                    })
                    seen_cores.add(core_key)
        
        # Filter groups by size
        valid_groups = {k: v for k, v in core_to_compounds.items() if len(v) >= min_pairs_per_core}
        
        if not valid_groups:
            st.warning(f"No valid cores found with {min_pairs_per_core}+ compounds.")
            return None, None
        
        # Step 3: Generate pairs
        status_text.text("Step 3/4: Generating molecular pairs...")
        progress_bar.progress(60)
        
        all_pairs = []
        
        for core, compounds in valid_groups.items():
            for i, j in combinations(range(len(compounds)), 2):
                c1 = compounds[i]
                c2 = compounds[j]
                
                # Calculate delta
                delta = c2['pIC50'] - c1['pIC50']
                
                # Create transform signature
                r1_smi = c1['rgroup_smiles'].replace('*', '*-')
                r2_smi = c2['rgroup_smiles'].replace('*', '*-')
                transform = f"{r1_smi}>>{r2_smi}"
                
                all_pairs.append({
                    'Core_SMILES': core,
                    'Compound1_Name': c1['name'],
                    'Compound1_pIC50': c1['pIC50'],
                    'Compound2_Name': c2['name'],
                    'Compound2_pIC50': c2['pIC50'],
                    'Transform': transform,
                    'Delta_pIC50': delta
                })
        
        pairs_df = pd.DataFrame(all_pairs)
        
        # Step 4: Analyze transforms
        status_text.text("Step 4/4: analyzing statistics...")
        progress_bar.progress(90)
        
        if pairs_df.empty:
            return None, None
            
        transform_data = []
        for transform, group in pairs_df.groupby('Transform'):
            count = len(group)
            if count >= min_transform_occurrence:
                deltas = group['Delta_pIC50'].values
                transform_data.append({
                    'Transform': transform,
                    'Count': count,
                    'Mean_ΔpIC50': np.mean(deltas),
                    'Std_ΔpIC50': np.std(deltas),
                    'Deltas': list(deltas)
                })
        
        transforms_df = pd.DataFrame(transform_data).sort_values('Count', ascending=False) if transform_data else None
        
        progress_bar.progress(100)
        status_text.empty()
        
        return pairs_df, transforms_df

    def rxn_to_base64_image(transform_smiles):
        """Convert transform SMILES to base64 image"""
        try:
            parts = transform_smiles.split('>>')
            if len(parts) != 2: return None
            
            mol1 = Chem.MolFromSmiles(parts[0].replace('*-', '[*]'))
            mol2 = Chem.MolFromSmiles(parts[1].replace('*-', '[*]'))
            
            if mol1 and mol2:
                img = Draw.MolsToGridImage([mol1, mol2], molsPerRow=2, subImgSize=(150, 100))
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                return f'<img src="data:image/png;base64,{img_str}" class="mmp-image">'
        except:
            return None

    def stripplot_base64_image(deltas):
        """Create strip plot of delta values"""
        try:
            fig, ax = plt.subplots(figsize=(4, 2))
            y = np.random.normal(0, 0.02, len(deltas))
            ax.scatter(deltas, y, alpha=0.6, s=30)
            ax.axvline(np.mean(deltas), color='red', linestyle='--')
            ax.axvline(0, color='black', linestyle='-', alpha=0.3)
            ax.set_yticks([])
            plt.tight_layout()
            
            buf = BytesIO()
            plt.savefig(buf, format="png", dpi=100)
            plt.close(fig)
            img_str = base64.b64encode(buf.getvalue()).decode()
            return f'<img src="data:image/png;base64,{img_str}" class="mmp-image">'
        except:
            return None

# -----------------------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------------------

if RDKIT_AVAILABLE:
    # Sidebar
    with st.sidebar:
        st.markdown("## 📋 Configuration")
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        st.markdown("### ⚙️ Analysis Parameters")
        min_pairs_per_core = st.slider("Min compounds per core", 2, 10, 3)
        min_transform_occurrence = st.slider("Min transform occurrences", 1, 20, 2)
        
        st.info("ℹ️ Molecular Weight and Property filters have been removed. All valid molecules are processed.")

    # Main Area
    if uploaded_file is None:
        st.info("Please upload a CSV file in the sidebar to begin.")
        
        # Example data generation for quick start
        if st.button("Load Example Data"):
            data = {
                'SMILES': [
                    'Cc1ccccc1C(=O)O', 'Cc1ccccc1C(=O)N', 'Cc1ccccc1C(=O)OC', 
                    'Oc1ccccc1C(=O)O', 'Oc1ccccc1C(=O)N', 'Oc1ccccc1C(=O)OC'
                ],
                'pIC50': [5.1, 5.5, 4.8, 5.3, 5.7, 4.9],
                'Name': [f'Mol_{i}' for i in range(6)]
            }
            df_example = pd.DataFrame(data)
            csv = df_example.to_csv(index=False).encode('utf-8')
            st.download_button("Download Example CSV", csv, "example.csv", "text/csv")

    else:
        # Load Data
        df = load_and_preprocess_data(uploaded_file)
        
        if df is not None:
            st.markdown('<h2 class="section-header">📊 Dataset Overview</h2>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Compounds", len(df))
            c2.metric("Max MW", f"{df['MW'].max():.1f}")
            c3.metric("Avg pIC50", f"{df['pIC50'].mean():.2f}")
            
            if st.button("🚀 Run Analysis", type="primary"):
                pairs_df, transforms_df = perform_mmp_analysis_single_cut(
                    df, min_pairs_per_core, min_transform_occurrence
                )
                
                if pairs_df is not None:
                    st.success(f"Generated {len(pairs_df)} pairs from {len(pairs_df['Core_SMILES'].unique())} unique cores.")
                    
                    # Top Transforms
                    if transforms_df is not None:
                        st.subheader("Significant Structural Transforms")
                        
                        # Add visuals
                        transforms_df['Visualization'] = transforms_df['Transform'].apply(rxn_to_base64_image)
                        transforms_df['Distribution'] = transforms_df['Deltas'].apply(stripplot_base64_image)
                        
                        for _, row in transforms_df.head(20).iterrows():
                            with st.container():
                                c1, c2, c3 = st.columns([2, 1, 2])
                                with c1:
                                    st.code(row['Transform'])
                                    if row['Visualization']:
                                        st.markdown(row['Visualization'], unsafe_allow_html=True)
                                with c2:
                                    st.metric("Count", row['Count'])
                                    st.metric("Mean Δ", f"{row['Mean_ΔpIC50']:.2f}")
                                with c3:
                                    st.write("ΔpIC50 Distribution")
                                    if row['Distribution']:
                                        st.markdown(row['Distribution'], unsafe_allow_html=True)
                                st.divider()
                        
                        # Downloads
                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button(
                                "📥 Download Pairs (CSV)", 
                                pairs_df.to_csv(index=False), 
                                "mmp_pairs.csv"
                            )
                        with col2:
                            # Remove complex objects for CSV export
                            export_transforms = transforms_df.drop(['Visualization', 'Distribution', 'Deltas'], axis=1, errors='ignore')
                            st.download_button(
                                "📥 Download Transforms (CSV)", 
                                export_transforms.to_csv(index=False), 
                                "mmp_transforms.csv"
                            )
