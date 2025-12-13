import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import tempfile
import base64
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="MMP Analysis Tool",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .stProgress .st-bo {
        background-color: #4CAF50;
    }
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1E40AF;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .transform-card {
        background-color: #F0F9FF;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #93C5FD;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🧪 Matched Molecular Pairs Analysis</h1>', unsafe_allow_html=True)
st.markdown("""
This application performs Matched Molecular Pair (MMP) analysis to identify structural transformations 
that lead to changes in biological activity (pIC50). Upload your dataset containing SMILES, compound names, 
and pIC50 values to get started.
""")

# Sidebar
with st.sidebar:
    st.header("📁 Data Upload")
    
    uploaded_file = st.file_uploader(
        "Upload your CSV file", 
        type=['csv'],
        help="CSV should contain columns: 'SMILES', 'Name', 'pIC50'"
    )
    
    st.header("⚙️ Analysis Parameters")
    
    min_transform_occurrence = st.slider(
        "Minimum transform occurrence",
        min_value=2,
        max_value=20,
        value=5,
        help="Minimum number of times a transformation must occur to be included in analysis"
    )
    
    max_cuts = st.slider(
        "Maximum cuts for fragmentation",
        min_value=1,
        max_value=3,
        value=1,
        help="Maximum number of cuts to make when fragmenting molecules"
    )
    
    analysis_button = st.button(
        "🚀 Run MMP Analysis",
        type="primary",
        use_container_width=True
    )
    
    st.markdown("---")
    st.markdown("### 📚 References")
    st.markdown("""
    - Hussain & Rea (2010) *J Chem Inf Model*
    - Dossetter et al. (2013) *Drug Discov Today*
    - Wassermann et al. (2012) *Drug Dev Res*
    - Tyrchan & Evertsson (2017) *Comput Struct Biotechnol J*
    """)

# Function to display molecule images
def render_molecule(smiles, width=300, height=200):
    """Render a molecule from SMILES string"""
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        from PIL import Image
        
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            img = Draw.MolToImage(mol, size=(width, height))
            return img
        return None
    except:
        return None

# Function to display reaction
def render_reaction(reaction_smarts, width=400, height=150):
    """Render a reaction from SMARTS string"""
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        from rdkit.Chem import AllChem
        
        rxn = AllChem.ReactionFromSmarts(reaction_smarts, useSmiles=True)
        img = Draw.ReactionToImage(rxn, subImgSize=(width, height))
        return img
    except Exception as e:
        st.error(f"Error rendering reaction: {str(e)}")
        return None

# Main application logic
def main():
    if not uploaded_file and not analysis_button:
        # Show example data and instructions
        st.info("👈 Please upload a CSV file in the sidebar to begin analysis")
        
        # Show example data format
        example_data = pd.DataFrame({
            'Name': ['CHEMBL4585243', 'CHEMBL1935285', 'CHEMBL1935286'],
            'SMILES': [
                'O=C(O)[C@H]1[C@H](Cn2nnc3ccccc3c2=O)CC[C@@H]1C(=O)c1ccc(OCCC2CCOCC2)cc1',
                'Cc1ccc(-c2ccc3oc4cc(S(=O)(=O)N[C@H](C(=O)O)C(C)C)ccc4c3c2)o1',
                'CC(C)[C@H](NS(=O)(=O)c1ccc2c(c1)oc1ccc(-c3ccc(Cl)o3)cc12)C(=O)O'
            ],
            'pIC50': [10.07, 10.00, 10.00]
        })
        
        with st.expander("📋 Example Data Format"):
            st.dataframe(example_data)
            st.download_button(
                label="📥 Download Example CSV",
                data=example_data.to_csv(index=False).encode('utf-8'),
                file_name="example_mmp_data.csv",
                mime="text/csv"
            )
        
        return
    
    if uploaded_file and analysis_button:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            
            # Check required columns
            required_cols = ['SMILES', 'Name', 'pIC50']
            if not all(col in df.columns for col in required_cols):
                st.error(f"CSV must contain columns: {required_cols}")
                st.write("Columns found:", list(df.columns))
                return
            
            # Display dataset info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Compounds", len(df))
            with col2:
                st.metric("pIC50 Range", f"{df['pIC50'].min():.2f} - {df['pIC50'].max():.2f}")
            with col3:
                st.metric("Average pIC50", f"{df['pIC50'].mean():.2f}")
            
            # Show data preview
            with st.expander("🔍 View Uploaded Data"):
                st.dataframe(df.head())
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Import required libraries
            status_text.text("Step 1/5: Loading libraries...")
            progress_bar.progress(10)
            
            try:
                # Import all required libraries
                import pandas as pd
                from rdkit import Chem
                from operator import itemgetter
                from itertools import combinations
                from rdkit.Chem import AllChem
                import seaborn as sns
                import matplotlib.pyplot as plt
                import useful_rdkit_utils as uru
                
                # Import the scaffold_finder module
                import sys
                import os
                
                # Create a temporary file for scaffold_finder
                scaffold_code = '''import sys
from rdkit import Chem
from rdkit.Chem.rdMMPA import FragmentMol
from rdkit.Chem.rdRGroupDecomposition import RGroupDecompose
import itertools
import useful_rdkit_utils as uru
import pandas as pd
from tqdm.auto import tqdm
from glob import glob
import numpy as np


def cleanup_fragment(mol):
    """
    Replace atom map numbers with Hydrogens
    :param mol: input molecule
    :return: modified molecule, number of R-groups
    """
    rgroup_count = 0
    for atm in mol.GetAtoms():
        atm.SetAtomMapNum(0)
        if atm.GetAtomicNum() == 0:
            rgroup_count += 1
            atm.SetAtomicNum(1)
    mol = Chem.RemoveAllHs(mol)
    return mol, rgroup_count


def generate_fragments(mol):
    """
    Generate fragments using the RDKit
    :param mol: RDKit molecule
    :return: a Pandas dataframe with Scaffold SMILES, Number of Atoms, Number of R-Groups
    """
    # Generate molecule fragments
    frag_list = FragmentMol(mol)
    # Flatten the output into a single list
    flat_frag_list = [x for x in itertools.chain(*frag_list) if x]
    # The output of Fragment mol is contained in single molecules.  Extract the largest fragment from each molecule
    flat_frag_list = [uru.get_largest_fragment(x) for x in flat_frag_list]
    # Keep fragments where the number of atoms in the fragment is at least 2/3 of the number fragments in
    # input molecule
    num_mol_atoms = mol.GetNumAtoms()
    flat_frag_list = [x for x in flat_frag_list if x.GetNumAtoms() / num_mol_atoms > 0.67]
    # remove atom map numbers from the fragments
    flat_frag_list = [cleanup_fragment(x) for x in flat_frag_list]
    # Convert fragments to SMILES
    frag_smiles_list = [[Chem.MolToSmiles(x), x.GetNumAtoms(), y] for (x, y) in flat_frag_list]
    # Add the input molecule to the fragment list
    frag_smiles_list.append([Chem.MolToSmiles(mol), mol.GetNumAtoms(), 1])
    # Put the results into a Pandas dataframe
    frag_df = pd.DataFrame(frag_smiles_list, columns=["Scaffold", "NumAtoms", "NumRgroupgs"])
    # Remove duplicate fragments
    frag_df = frag_df.drop_duplicates("Scaffold")
    return frag_df


def find_scaffolds(df_in):
    """
    Generate scaffolds for a set of molecules
    :param df_in: Pandas dataframe with [SMILES, Name, RDKit molecule] columns
    :return: dataframe with molecules and scaffolds, dataframe with unique scaffolds
    """
    # Loop over molecules and generate fragments, fragments for each molecule are returned as a Pandas dataframe
    df_list = []
    for smiles, name, mol in tqdm(df_in[["SMILES", "Name", "mol"]].values):
        tmp_df = generate_fragments(mol).copy()
        tmp_df['Name'] = name
        tmp_df['SMILES'] = smiles
        df_list.append(tmp_df)
    # Combine the list of dataframes into a single dataframe
    mol_df = pd.concat(df_list)
    # Collect scaffolds
    scaffold_list = []
    for k, v in mol_df.groupby("Scaffold"):
        scaffold_list.append([k, len(v.Name.unique()), v.NumAtoms.values[0]])
    scaffold_df = pd.DataFrame(scaffold_list, columns=["Scaffold", "Count", "NumAtoms"])
    # Any fragment that occurs more times than the number of fragments can't be a scaffold
    num_df_rows = len(df_in)
    scaffold_df = scaffold_df.query("Count <= @num_df_rows")
    # Sort scaffolds by frequency
    scaffold_df = scaffold_df.sort_values(["Count", "NumAtoms"], ascending=[False, False])
    return mol_df, scaffold_df


def get_molecules_with_scaffold(scaffold, mol_df, activity_df):
    """
    Associate molecules with scaffolds
    :param scaffold: scaffold SMILES
    :param mol_df: dataframe with molecules and scaffolds, returned by find_scaffolds()
    :param activity_df: dataframe with [SMILES, Name, pIC50] columns
    :return: list of core(s) with R-groups labeled, dataframe with [SMILES, Name, pIC50]
    """
    match_df = mol_df.query("Scaffold == @scaffold")
    merge_df = match_df.merge(activity_df, on=["SMILES", "Name"])
    scaffold_mol = Chem.MolFromSmiles(scaffold)
    rgroup_match, rgroup_miss = RGroupDecompose(scaffold_mol, merge_df.mol, asSmiles=True)
    if len(rgroup_match):
        rgroup_df = pd.DataFrame(rgroup_match)
        return rgroup_df.Core.unique(), merge_df[["SMILES", "Name", "pIC50"]]
    else:
        return [], merge_df[["SMILES", "Name", "pIC50"]]


def main():
    """
    Read all SMILES files in the current directory, generate scaffolds, report stats for each scaffold
    :return: None
    """
    filename_list = glob("CHEMBL237.smi")
    out_list = []
    for filename in filename_list:
        print(filename)
        input_df = pd.read_csv(filename, names=["SMILES", "Name", "pIC50"])
        input_df['mol'] = input_df.SMILES.apply(Chem.MolFromSmiles)
        input_df['SMILES'] = input_df.mol.apply(Chem.MolToSmiles)
        mol_df, scaffold_df = find_scaffolds(input_df)
        scaffold_1 = scaffold_df.Scaffold.values[0]
        scaffold_list, scaffold_mol_df = get_molecules_with_scaffold(scaffold_1, mol_df, input_df)
        ic50_list = scaffold_mol_df.pIC50
        if len(scaffold_list):
            print(scaffold_list[0], len(scaffold_mol_df), max(ic50_list) - min(ic50_list), np.std(ic50_list))
            out_list.append([filename, scaffold_list[0], len(scaffold_mol_df), max(ic50_list) - min(ic50_list),
                             np.std(ic50_list)])
        else:
            print(f"Could not find a scaffold for {filename}", file=sys.stderr)
    out_df = pd.DataFrame(out_list, columns=["Filename", "Scaffold", "Count", "Range", "Std"])
    out_df.to_csv("scaffold_stats.csv", index=False)


if __name__ == "__main__":
    main()'''
                
                # Write scaffold_finder to a temporary file
                temp_dir = tempfile.mkdtemp()
                scaffold_path = os.path.join(temp_dir, "scaffold_finder.py")
                with open(scaffold_path, "w") as f:
                    f.write(scaffold_code)
                
                # Add to Python path
                sys.path.insert(0, temp_dir)
                
                # Now import the module
                from scaffold_finder import FragmentMol, generate_fragments, find_scaffolds, get_molecules_with_scaffold
                
            except ImportError as e:
                st.error(f"Failed to import required libraries: {str(e)}")
                st.info("Please install required packages: pip install pandas rdkit useful_rdkit_utils seaborn matplotlib")
                return
            
            # Step 2: Prepare molecules
            status_text.text("Step 2/5: Preparing molecules...")
            progress_bar.progress(30)
            
            # Add molecule column
            df['mol'] = df.SMILES.apply(Chem.MolFromSmiles)
            
            # Function to remove map numbers
            def remove_map_nums(mol):
                for atm in mol.GetAtoms():
                    atm.SetAtomMapNum(0)
                return mol
            
            # Function to sort fragments
            def sort_fragments(mol):
                frag_list = list(Chem.GetMolFrags(mol, asMols=True))
                frag_list = [remove_map_nums(x) for x in frag_list]
                frag_num_atoms_list = [(x.GetNumAtoms(), x) for x in frag_list]
                frag_num_atoms_list.sort(key=itemgetter(0), reverse=True)
                return [x[1] for x in frag_num_atoms_list]
            
            # Step 3: Fragment molecules
            status_text.text("Step 3/5: Fragmenting molecules...")
            progress_bar.progress(50)
            
            row_list = []
            for smiles, name, pIC50, mol in df.values:
                # Use FragmentMol from scaffold_finder
                frag_list = FragmentMol(mol, maxCuts=max_cuts)
                for _, frag_mol in frag_list:
                    pair_list = sort_fragments(frag_mol)
                    if len(pair_list) == 2:  # We expect exactly 2 fragments (Core and R-group)
                        tmp_list = [smiles] + [Chem.MolToSmiles(x) for x in pair_list] + [name, pIC50]
                        row_list.append(tmp_list)
            
            row_df = pd.DataFrame(row_list, columns=["SMILES", "Core", "R_group", "Name", "pIC50"])
            
            # Step 4: Find matched pairs
            status_text.text("Step 4/5: Finding matched pairs...")
            progress_bar.progress(70)
            
            delta_list = []
            for k, v in row_df.groupby("Core"):
                if len(v) > 2:
                    for a, b in combinations(range(0, len(v)), 2):
                        reagent_a = v.iloc[a]
                        reagent_b = v.iloc[b]
                        if reagent_a.SMILES == reagent_b.SMILES:
                            continue
                        reagent_a, reagent_b = sorted([reagent_a, reagent_b], key=lambda x: x.SMILES)
                        delta = reagent_b.pIC50 - reagent_a.pIC50
                        transform_str = f"{reagent_a.R_group.replace('*','*-')}>>{reagent_b.R_group.replace('*','*-')}"
                        delta_list.append(list(reagent_a.values) + list(reagent_b.values) + [transform_str, delta])
            
            cols = ["SMILES_1", "Core_1", "R_group_1", "Name_1", "pIC50_1",
                   "SMILES_2", "Core_2", "Rgroup_2", "Name_2", "pIC50_2",
                   "Transform", "Delta"]
            
            delta_df = pd.DataFrame(delta_list, columns=cols)
            
            # Step 5: Analyze frequent transforms
            status_text.text("Step 5/5: Analyzing frequent transforms...")
            progress_bar.progress(90)
            
            mmp_list = []
            for k, v in delta_df.groupby("Transform"):
                if len(v) >= min_transform_occurrence:
                    mmp_list.append([k, len(v), v.Delta.values])
            
            mmp_df = pd.DataFrame(mmp_list, columns=["Transform", "Count", "Deltas"])
            mmp_df['mean_delta'] = [x.mean() for x in mmp_df.Deltas]
            mmp_df['rxn_mol'] = mmp_df.Transform.apply(
                lambda x: AllChem.ReactionFromSmarts(x, useSmiles=True)
            )
            mmp_df = mmp_df.sort_values("mean_delta", ascending=False)
            
            # Create index mapping
            transform_dict = dict([(a, b) for a, b in mmp_df[["Transform", "Count"]].values])
            delta_df['transform_count'] = [transform_dict.get(x, 0) for x in delta_df.Transform]
            
            progress_bar.progress(100)
            status_text.text("Analysis complete!")
            
            # Display Results
            st.markdown('<h2 class="sub-header">📊 Analysis Results</h2>', unsafe_allow_html=True)
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Fragments", len(row_df))
            with col2:
                st.metric("Unique Cores", row_df["Core"].nunique())
            with col3:
                st.metric("Matched Pairs", len(delta_df))
            with col4:
                st.metric("Frequent Transforms", len(mmp_df))
            
            # Display frequent transforms
            if len(mmp_df) > 0:
                st.markdown('<h3 class="sub-header">🔬 Frequent Transformations</h3>', unsafe_allow_html=True)
                
                for idx, row in mmp_df.iterrows():
                    with st.container():
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.markdown(f"""
                            <div class="transform-card">
                                <strong>Transform:</strong> {row['Transform']}<br>
                                <strong>Count:</strong> {row['Count']}<br>
                                <strong>Mean ΔpIC50:</strong> {row['mean_delta']:.3f}<br>
                                <strong>Std Dev:</strong> {row['Deltas'].std():.3f}
                            </div>
                            """, unsafe_allow_html=True)
                        with col2:
                            img = render_reaction(row['Transform'])
                            if img:
                                st.image(img, caption="Reaction", use_column_width=True)
                
                # Show top improvements
                st.markdown('<h3 class="sub-header">🏆 Top Improvements</h3>', unsafe_allow_html=True)
                
                top_improvements = mmp_df.sort_values('mean_delta', ascending=False).head(3)
                
                for i, (_, row) in enumerate(top_improvements.iterrows(), 1):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.markdown(f"**Rank {i}**")
                        img = render_reaction(row['Transform'])
                        if img:
                            st.image(img, caption=row['Transform'][:50] + "...", use_column_width=True)
                    with col2:
                        st.markdown(f"""
                        **Transform:** `{row['Transform']}`  
                        **Mean ΔpIC50:** `{row['mean_delta']:.3f}`  
                        **Occurrences:** `{row['Count']}`  
                        **Range:** `{row['Deltas'].min():.3f} to {row['Deltas'].max():.3f}`  
                        **Standard Deviation:** `{row['Deltas'].std():.3f}`
                        """)
                
                # Show distribution of deltas
                st.markdown('<h3 class="sub-header">📈 ΔpIC50 Distribution</h3>', unsafe_allow_html=True)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                all_deltas = []
                for deltas in mmp_df['Deltas']:
                    all_deltas.extend(deltas)
                
                ax.hist(all_deltas, bins=30, edgecolor='black', alpha=0.7)
                ax.axvline(x=0, color='red', linestyle='--', label='ΔpIC50 = 0')
                ax.set_xlabel('ΔpIC50')
                ax.set_ylabel('Frequency')
                ax.set_title('Distribution of Activity Changes')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                
                # Download buttons
                st.markdown('<h3 class="sub-header">💾 Download Results</h3>', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    csv = row_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Fragments (CSV)",
                        data=csv,
                        file_name="molecule_fragments.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    csv = delta_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Matched Pairs (CSV)",
                        data=csv,
                        file_name="matched_pairs.csv",
                        mime="text/csv"
                    )
                
                with col3:
                    csv = mmp_df.drop('rxn_mol', axis=1).to_csv(index=False)
                    st.download_button(
                        label="📥 Download Transforms (CSV)",
                        data=csv,
                        file_name="frequent_transforms.csv",
                        mime="text/csv"
                    )
                
                # Show detailed data in expanders
                with st.expander("🔍 View Fragmented Molecules"):
                    st.dataframe(row_df.head(100))
                
                with st.expander("🔍 View All Matched Pairs"):
                    st.dataframe(delta_df.head(100))
                
                with st.expander("🔍 View All Transforms"):
                    display_df = mmp_df.drop('rxn_mol', axis=1).copy()
                    display_df['Delta_Values'] = display_df['Deltas'].apply(lambda x: str(list(x.round(3))))
                    st.dataframe(display_df)
            
            else:
                st.warning(f"No transformations found with minimum occurrence of {min_transform_occurrence}. Try lowering the threshold.")
                
        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main()
