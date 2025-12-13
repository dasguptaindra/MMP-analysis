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
.main-header { font-size: 2.5rem; color: #1E3A8A; text-align: center; margin-bottom: 2rem; }
.section-header { font-size: 1.8rem; color: #2563EB; margin-top: 2rem; margin-bottom: 1rem; border-bottom: 2px solid #E5E7EB; padding-bottom: 0.5rem; }
.transform-card { background-color: #F8FAFC; border-radius: 10px; padding: 1.5rem; margin-bottom: 1.5rem; border-left: 4px solid #3B82F6; }
.metric-card { background-color: #F0F9FF; border-radius: 8px; padding: 1rem; text-align: center; margin: 0.5rem; }
.warning-box { background-color: #FEF3C7; border-left: 4px solid #F59E0B; padding: 1rem; border-radius: 5px; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🧪 Matched Molecular Pair (MMP) Analysis Tool</h1>', unsafe_allow_html=True)

# RDKit import
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw
    RDKIT_AVAILABLE = True
except ImportError as e:
    st.error(f"RDKit not available: {e}")
    RDKIT_AVAILABLE = False

# Sidebar
with st.sidebar:
    st.markdown("## 📋 Configuration")
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

    if uploaded_file and RDKIT_AVAILABLE:
        st.markdown("### ⚙️ Parameters")
        min_occurrence = st.slider("Minimum transform occurrences", 1, 20, 5)
        st.markdown("### 🧹 Molecule Cleaning")
        sanitize_molecules = st.checkbox("Sanitize molecules", value=True)
        kekulize_molecules = st.checkbox("Kekulize molecules", value=False)
        st.markdown("### 👀 Display Options")
        show_all_transforms = st.checkbox("Show all transformations", value=False)
        transforms_to_display = st.slider("Number of transforms to display", 1, 50, 10, disabled=show_all_transforms)
        st.markdown("### 🔬 Analysis")
        show_top_positive = st.checkbox("Show top positive transforms", value=True)
        show_top_negative = st.checkbox("Show top negative transforms", value=True)
        show_compound_examples = st.checkbox("Show compound examples", value=True)
        st.markdown("### 🔗 Pair Generation Logic")
        st.info("Pairs are generated only when 3+ compounds share the same core.")
        st.markdown("### 🐛 Debug")
        show_debug_info = st.checkbox("Show debug information", value=False)
        st.markdown("### 💾 Export")
        save_results = st.checkbox("Save results to Excel")

# Helper functions
if RDKIT_AVAILABLE:
    @st.cache_data
    def load_data(file, sanitize=True, kekulize=False):
        df = pd.read_csv(file)
        required_cols = ['SMILES', 'pIC50']
        if not all(col in df.columns for col in required_cols):
            st.error(f"CSV must contain columns: {required_cols}")
            return None

        molecules, errors = [], []
        for idx, smiles in enumerate(df['SMILES']):
            try:
                mol = Chem.MolFromSmiles(str(smiles))
                if mol is None:
                    errors.append(f"Row {idx}: Invalid SMILES '{smiles}'")
                    molecules.append(None)
                    continue
                if sanitize:
                    try: Chem.SanitizeMol(mol)
                    except: pass
                if kekulize:
                    try: Chem.Kekulize(mol, clearAromaticFlags=True)
                    except: pass
                frags = Chem.GetMolFrags(mol, asMols=True)
                if frags: mol = max(frags, key=lambda x: x.GetNumAtoms())
                molecules.append(mol)
            except Exception as e:
                errors.append(f"Row {idx}: Error processing '{smiles}' - {str(e)}")
                molecules.append(None)

        df['mol'] = molecules
        valid_df = df[df['mol'].notna()].copy()
        if len(valid_df) < len(df):
            st.warning(f"Removed {len(df) - len(valid_df)} rows with invalid molecules")
        return valid_df

    def remove_map_nums(mol):
        if mol is None: return None
        for atm in mol.GetAtoms(): atm.SetAtomMapNum(0)
        return mol

    def sort_fragments(mol):
        if mol is None: return []
        try:
            frag_list = list(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False))
            frag_list = [remove_map_nums(x) for x in frag_list]
            frag_num_atoms_list = [(x.GetNumAtoms(), x) for x in frag_list]
            frag_num_atoms_list.sort(key=itemgetter(0), reverse=True)
            return [x[1] for x in frag_num_atoms_list]
        except: return []

    def FragmentMol(mol, maxCuts=1):
        results = []
        try:
            mol_copy = Chem.Mol(mol)
            for bond in mol_copy.GetBonds():
                if bond.GetBondType() == Chem.BondType.SINGLE:
                    try:
                        emol = Chem.EditableMol(mol_copy)
                        emol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                        frag_mol = emol.GetMol()
                        try: Chem.SanitizeMol(frag_mol)
                        except: pass
                        results.append((f"CUT_{bond.GetIdx()}", frag_mol))
                    except: continue
        except: pass
        if not results: results.append(("NO_CUT", mol))
        return results

    def perform_mmp_analysis(df, min_transform_occurrence, show_debug=False):
        if df is None or len(df) == 0: return None, None
        row_list = []
        for idx, row in df.iterrows():
            mol = row['mol']
            if mol is None: continue
            frag_list = FragmentMol(mol, maxCuts=1)
            for _, frag_mol in frag_list:
                pair_list = sort_fragments(frag_mol)
                if len(pair_list) >= 2:
                    try:
                        core_smiles = Chem.MolToSmiles(pair_list[0])
                        rgroup_smiles = Chem.MolToSmiles(pair_list[1])
                        row_list.append([row['SMILES'], core_smiles, rgroup_smiles, row.get('Name', f"CMPD_{idx}"), row['pIC50']])
                    except: continue
        if not row_list: return None, None
        row_df = pd.DataFrame(row_list, columns=["SMILES", "Core", "R_group", "Name", "pIC50"])
        delta_list = []
        for k, v in row_df.groupby("Core"):
            if len(v) > 2:
                for a, b in combinations(range(len(v)), 2):
                    reagent_a = v.iloc[a]
                    reagent_b = v.iloc[b]
                    if reagent_a.SMILES == reagent_b.SMILES: continue
                    reagent_a, reagent_b = sorted([reagent_a, reagent_b], key=lambda x: x.SMILES)
                    delta = reagent_b.pIC50 - reagent_a.pIC50
                    transform_str = f"{reagent_a.R_group.replace('*','*-')}>>{reagent_b.R_group.replace('*','*-')}"
                    delta_list.append([
                        reagent_a.SMILES, reagent_a.Core, reagent_a.R_group, reagent_a.Name, reagent_a.pIC50,
                        reagent_b.SMILES, reagent_b.Core, reagent_b.R_group, reagent_b.Name, reagent_b.pIC50,
                        transform_str, delta
                    ])
        if not delta_list: return row_df, None
        cols = [
            "SMILES_1","Core_1","R_group_1","Name_1","pIC50_1",
            "SMILES_2","Core_2","R_group_2","Name_2","pIC50_2",
            "Transform","Delta"
        ]
        delta_df = pd.DataFrame(delta_list, columns=cols)
        mmp_list = []
        for k, v in delta_df.groupby("Transform"):
            if len(v) >= min_transform_occurrence:
                mmp_list.append([k, len(v), v.Delta.values])
        if not mmp_list: return delta_df, None
        mmp_df = pd.DataFrame(mmp_list, columns=["Transform","Count","Deltas"])
        mmp_df['mean_delta'] = [x.mean() for x in mmp_df.Deltas]
        rxn_mols = []
        for transform in mmp_df['Transform']:
            try: rxn = AllChem.ReactionFromSmarts(transform.replace('*-','*'), useSmiles=True); rxn_mols.append(rxn)
            except: rxn_mols.append(None)
        mmp_df['rxn_mol'] = rxn_mols
        return delta_df, mmp_df

    def plot_stripplot_to_fig(deltas):
        fig, ax = plt.subplots(figsize=(4,1.5))
        sns.stripplot(x=deltas, ax=ax, jitter=0.2, alpha=0.7, s=5, color='blue')
        ax.axvline(0, ls='--', c='red')
        if deltas:
            data_min, data_max = min(deltas), max(deltas)
            padding = max(0.5, (data_max-data_min)*0.1)
            ax.set_xlim(data_min-padding, data_max+padding)
        else: ax.set_xlim(-5,5)
        ax.set_xlabel('ΔpIC50'); ax.set_yticks([])
        plt.tight_layout()
        return fig

    def get_rxn_image(rxn_mol):
        if rxn_mol is None: return None
        try:
            img = Draw.ReactionToImage(rxn_mol, subImgSize=(300,150))
            buffered = io.BytesIO(); img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode()
        except: return None

    def find_examples(delta_df, transform):
        examples = delta_df[delta_df['Transform']==transform]
        if len(examples)==0: return None
        example_list=[]
        for _, row in examples.sort_values("Delta", ascending=False).iterrows():
            example_list.append({"SMILES": row['SMILES_1'], "Name": row['Name_1'], "pIC50": row['pIC50_1'], "Type": "Before"})
            example_list.append({"SMILES": row['SMILES_2'], "Name": row['Name_2'], "pIC50": row['pIC50_2'], "Type": "After"})
        return pd.DataFrame(example_list)

# Main App Logic
if not RDKIT_AVAILABLE:
    st.error("RDKit is required. Please install rdkit-pypi or use conda installation.")
elif uploaded_file is not None:
    df = load_data(uploaded_file, sanitize=sanitize_molecules, kekulize=kekulize_molecules)
    if df is not None and len(df)>0:
        st.markdown('<h2 class="section-header">🔍 MMP Analysis Results</h2>', unsafe_allow_html=True)
        delta_df, mmp_df = perform_mmp_analysis(df, min_occurrence, show_debug=show_debug_info)
        if delta_df is not None:
            st.success("Analysis complete!")
            st.metric("Total Pairs Generated", len(delta_df))
            if mmp_df is not None:
                st.metric("Unique Transforms", len(mmp_df))
                st.metric("Average Transform Frequency", f"{mmp_df['Count'].mean():.1f}")
else:
    st.info("Upload a CSV file with at least 'SMILES' and 'pIC50' columns to start analysis.")
