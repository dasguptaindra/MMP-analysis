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
    page_title="Advanced MMP Analysis (No Filters)",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for tables and layout
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1E3A8A; text-align: center; margin-bottom: 2rem; }
    .section-header { font-size: 1.5rem; color: #2563EB; margin-top: 1.5rem; border-bottom: 2px solid #E5E7EB; }
    table { width: 100%; border-collapse: collapse; }
    th { background-color: #f8f9fa; text-align: left; padding: 8px; border-bottom: 2px solid #dee2e6; }
    td { padding: 8px; border-bottom: 1px solid #dee2e6; vertical-align: middle; }
    .transform-code { font-family: monospace; background: #f1f3f5; padding: 2px 4px; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🧪 Advanced MMP Analysis Tool</h1>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# RDKIT IMPORTS
# -----------------------------------------------------------------------------
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, Descriptors
    from rdkit.Chem.rdMMPA import FragmentMol
    from rdkit.Chem import rdMolDraw2D
    RDKIT_AVAILABLE = True
except ImportError:
    st.error("❌ RDKit is not installed. Please run: pip install rdkit")
    RDKIT_AVAILABLE = False

# -----------------------------------------------------------------------------
# CORE LOGIC FUNCTIONS
# -----------------------------------------------------------------------------

if RDKIT_AVAILABLE:
    def remove_map_nums(mol):
        """Remove atom map numbers from a molecule to ensure canonical SMILES."""
        for atm in mol.GetAtoms():
            atm.SetAtomMapNum(0)
        return mol
    
    def sort_fragments(mol):
        """
        Takes a fragmented molecule and returns a list of fragments 
        sorted by size (largest = Core, smallest = R-group).
        """
        frag_list = list(Chem.GetMolFrags(mol, asMols=True))
        [remove_map_nums(x) for x in frag_list]
        # Sort by number of atoms (Descending)
        frag_num_atoms_list = [(x.GetNumAtoms(), x) for x in frag_list]
        frag_num_atoms_list.sort(key=itemgetter(0), reverse=True)
        return [x[1] for x in frag_num_atoms_list]
    
    def generate_fragments_exhaustive(mol):
        """
        Generate fragments using RDKit's FragmentMol with exhaustive single cuts.
        Returns a list of dictionaries containing Core/R-group info.
        """
        # maxCuts=1 ensures we get exactly one cut (2 pieces)
        frag_list = FragmentMol(mol, maxCuts=1)
        results = []
        
        for _, frag_mol in frag_list:
            pair_list = sort_fragments(frag_mol)
            
            # We strictly want 2 pieces: 1 Core (larger), 1 R-group (smaller)
            if len(pair_list) == 2:
                core_mol, rgroup_mol = pair_list[0], pair_list[1]
                
                core_smiles = Chem.MolToSmiles(core_mol)
                rgroup_smiles = Chem.MolToSmiles(rgroup_mol)
                
                # Verify attachment points exist in both parts
                if '*' in core_smiles and '*' in rgroup_smiles:
                    results.append({
                        'core_smiles': core_smiles,
                        'rgroup_smiles': rgroup_smiles,
                        'core_size': core_mol.GetNumAtoms()
                    })
        return results

    @st.cache_data
    def load_and_preprocess_data(file):
        """
        Load CSV and process molecules. 
        CRITICAL: No filtering of MW or properties is applied.
        """
        if file is None:
            return None
        
        try:
            df = pd.read_csv(file)
            
            # Normalize column names (case insensitive)
            cols = {c.lower(): c for c in df.columns}
            
            if 'smiles' not in cols:
                st.error("CSV must contain 'SMILES' column")
                return None
            
            # Rename to standard names
            df.rename(columns={cols['smiles']: 'SMILES'}, inplace=True)
            
            if 'pic50' in cols:
                df.rename(columns={cols['pic50']: 'pIC50'}, inplace=True)
            elif 'pIC50' not in df.columns:
                st.warning("⚠️ 'pIC50' column not found. Generating random values for testing.")
                np.random.seed(42)
                df['pIC50'] = np.random.uniform(4.0, 9.0, len(df))
            
            if 'Name' not in df.columns:
                if 'name' in cols:
                    df.rename(columns={cols['name']: 'Name'}, inplace=True)
                else:
                    df['Name'] = [f"Cmp_{i+1}" for i in range(len(df))]
            
            # Convert to RDKit Molecules
            molecules = []
            valid_indices = []
            
            for idx, row in df.iterrows():
                try:
                    smi = str(row['SMILES']).strip()
                    mol = Chem.MolFromSmiles(smi)
                    if mol is not None:
                        molecules.append(mol)
                        valid_indices.append(idx)
                except:
                    pass
            
            if not molecules:
                st.error("No valid molecules found in file.")
                return None
            
            # Rebuild DataFrame with only valid molecules
            final_df = df.iloc[valid_indices].copy()
            final_df['mol'] = molecules  # Store objects for calculation
            
            # Calculate MW just for info (not filtering)
            final_df['MW'] = [Descriptors.MolWt(m) for m in molecules]
            
            return final_df
            
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return None

    def perform_mmp_analysis(df, min_pairs, min_occurrence):
        """
        Main Analysis Workflow.
        1. Fragment all molecules.
        2. Group by Common Core.
        3. Generate Pairs.
        4. Calculate Stats.
        """
        
        # UI Progress
        prog_bar = st.progress(0)
        status = st.empty()
        
        # 1. Fragmentation
        status.text("Step 1/3: Fragmenting molecules...")
        compound_fragments = [] # List of dicts
        
        for idx, row in df.iterrows():
            mol = row['mol']
            frags = generate_fragments_exhaustive(mol)
            
            for f in frags:
                compound_fragments.append({
                    'id': row['Name'],
                    'smiles': row['SMILES'],
                    'pIC50': row['pIC50'],
                    'core': f['core_smiles'].replace('[*]', '*'), # Standardize wildcard
                    'rgroup': f['rgroup_smiles']
                })
        
        if not compound_fragments:
            st.error("Fragmentation failed.")
            return None, None

        prog_bar.progress(30)
        
        # 2. Grouping & Pairing
        status.text("Step 2/3: Grouping cores and finding pairs...")
        
        # Organize by core
        core_groups = defaultdict(list)
        for item in compound_fragments:
            core_groups[item['core']].append(item)
            
        pairs_data = []
        
        for core, group in core_groups.items():
            # We need at least 'min_pairs' compounds to form a valid series
            if len(group) < min_pairs:
                continue
                
            # Create pairs using combinations
            for c1, c2 in combinations(group, 2):
                # Calculate Delta
                delta = c2['pIC50'] - c1['pIC50']
                
                # Define Transform: R1 >> R2
                # Standardize direction (e.g., smaller SMILES string first) to deduplicate
                # But here we stick to the generated order and just ensure consistency
                r1 = c1['rgroup'].replace('*', '*-')
                r2 = c2['rgroup'].replace('*', '*-')
                transform = f"{r1}>>{r2}"
                
                pairs_data.append({
                    'Core': core,
                    'Compound_1': c1['id'],
                    'Compound_2': c2['id'],
                    'pIC50_1': c1['pIC50'],
                    'pIC50_2': c2['pIC50'],
                    'Delta_pIC50': delta,
                    'Transform': transform
                })
        
        pairs_df = pd.DataFrame(pairs_data)
        if pairs_df.empty:
            return None, None
            
        prog_bar.progress(60)

        # 3. Statistics
        status.text("Step 3/3: Calculating statistics...")
        
        stats_data = []
        for transform, sub_df in pairs_df.groupby('Transform'):
            count = len(sub_df)
            if count >= min_occurrence:
                deltas = sub_df['Delta_pIC50'].values
                stats_data.append({
                    'Transform': transform,
                    'Count': count,
                    'Mean_Delta': np.mean(deltas),
                    'Std_Delta': np.std(deltas),
                    'Deltas_List': list(deltas) # Stored for plotting
                })
        
        if not stats_data:
            return pairs_df, None
            
        stats_df = pd.DataFrame(stats_data).sort_values('Count', ascending=False)
        
        prog_bar.progress(100)
        status.empty()
        
        return pairs_df, stats_df

    # -----------------------------------------------------------------------------
    # VISUALIZATION FUNCTIONS
    # -----------------------------------------------------------------------------

    def render_transform_image(transform_str):
        """Generate base64 image of the chemical transformation."""
        try:
            parts = transform_str.split('>>')
            m1 = Chem.MolFromSmiles(parts[0].replace('*-', '[*]'))
            m2 = Chem.MolFromSmiles(parts[1].replace('*-', '[*]'))
            
            if m1 and m2:
                img = Draw.MolsToGridImage([m1, m2], molsPerRow=2, subImgSize=(150, 100))
                # Convert to base64
                buf = BytesIO()
                img.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode()
                return f'<img src="data:image/png;base64,{b64}" class="mmp-image"/>'
        except:
            return ""
        return ""

    def render_strip_plot(deltas):
        """Generate base64 image of the delta distribution."""
        try:
            fig, ax = plt.subplots(figsize=(3, 1.5))
            # Jitter y-values slightly for visibility
            y = np.random.normal(0, 0.05, len(deltas))
            
            ax.scatter(deltas, y, alpha=0.6, s=20, c='#2563EB')
            ax.axvline(np.mean(deltas), color='red', linestyle='--', linewidth=1)
            ax.axvline(0, color='black', linewidth=0.5)
            
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            
            plt.tight_layout()
            
            buf = BytesIO()
            plt.savefig(buf, format="PNG", dpi=100)
            plt.close(fig)
            b64 = base64.b64encode(buf.getvalue()).decode()
            return f'<img src="data:image/png;base64,{b64}" class="mmp-image"/>'
        except:
            return ""

# -----------------------------------------------------------------------------
# MAIN APPLICATION
# -----------------------------------------------------------------------------

if RDKIT_AVAILABLE:
    # --- Sidebar ---
    with st.sidebar:
        st.header("1. Upload Data")
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        st.caption("Required columns: SMILES, pIC50 (optional)")
        
        st.divider()
        st.header("2. Settings")
        min_compounds = st.slider("Min Compounds per Core", 2, 10, 3)
        min_occur = st.slider("Min Transform Occurrence", 1, 20, 2)

    # --- Main Content ---
    if uploaded_file:
        df = load_and_preprocess_data(uploaded_file)
        
        if df is not None:
            # Info Cards
            st.markdown('<h2 class="section-header">📂 Dataset Overview</h2>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Compounds", len(df))
            c2.metric("Avg pIC50", f"{df['pIC50'].mean():.2f}")
            c3.metric("Avg MW", f"{df['MW'].mean():.1f}")
            
            # Preview Data (HIDING THE RAW OBJECTS)
            with st.expander("View Raw Data Table"):
                # Drop the 'mol' column before displaying so users don't see <rdkit...> objects
                display_df = df.drop(columns=['mol'], errors='ignore')
                st.dataframe(display_df.head(10))
            
            # Run Analysis
            st.markdown('<h2 class="section-header">⚙️ Analysis</h2>', unsafe_allow_html=True)
            
            if st.button("Run MMP Analysis", type="primary"):
                pairs_df, stats_df = perform_mmp_analysis(df, min_compounds, min_occur)
                
                if pairs_df is not None:
                    # Results Summary
                    st.success(f"Analysis Complete: Found {len(pairs_df)} pairs.")
                    
                    if stats_df is not None:
                        st.subheader("Significant Transformations")
                        
                        # Prepare HTML Table
                        table_html = """
                        <table>
                            <thead>
                                <tr>
                                    <th>Transformation</th>
                                    <th>Structure</th>
                                    <th>Count</th>
                                    <th>Avg ΔpIC50</th>
                                    <th>Distribution</th>
                                </tr>
                            </thead>
                            <tbody>
                        """
                        
                        # Iterate through top results
                        for _, row in stats_df.head(50).iterrows():
                            transform_img = render_transform_image(row['Transform'])
                            dist_img = render_strip_plot(row['Deltas_List'])
                            
                            table_html += f"""
                                <tr>
                                    <td><div class="transform-code">{row['Transform']}</div></td>
                                    <td>{transform_img}</td>
                                    <td><strong>{row['Count']}</strong></td>
                                    <td>{row['Mean_Delta']:.2f}</td>
                                    <td>{dist_img}</td>
                                </tr>
                            """
                        
                        table_html += "</tbody></table>"
                        st.markdown(table_html, unsafe_allow_html=True)
                        
                        # Downloads
                        st.markdown("### 📥 Downloads")
                        c1, c2 = st.columns(2)
                        
                        # Pairs CSV
                        csv_pairs = pairs_df.to_csv(index=False).encode('utf-8')
                        c1.download_button("Download All Pairs (CSV)", csv_pairs, "mmp_pairs.csv", "text/csv")
                        
                        # Stats CSV (Remove list column for export)
                        csv_stats = stats_df.drop(columns=['Deltas_List']).to_csv(index=False).encode('utf-8')
                        c2.download_button("Download Transform Stats (CSV)", csv_stats, "mmp_stats.csv", "text/csv")
                        
                    else:
                        st.warning("Pairs found, but no transformation met the minimum occurrence threshold.")
                        st.dataframe(pairs_df.head())
                else:
                    st.error("No Matched Molecular Pairs found with current settings.")
                    
    else:
        # Welcome / Empty State
        st.info("👈 Please upload a CSV file in the sidebar to start.")
        st.markdown("""
        **CSV Format Guide:**
        - Must contain a `SMILES` column.
        - Should contain `pIC50` (or `Activity`).
        - Optional: `Name` or `ID`.
        """)
