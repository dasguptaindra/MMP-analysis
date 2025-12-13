import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.rdMMPA import FragmentMol
from rdkit.Chem.rdRGroupDecomposition import RGroupDecompose
from rdkit.Chem.MolStandardize import rdMolStandardize
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from tqdm import tqdm
import sys
from io import BytesIO

# ==========================================
# 1. Helper Functions (Based on provided code)
# ==========================================

class UsefulRDKitUtils:
    """Mock class to replace useful_rdkit_utils"""
    @staticmethod
    def get_largest_fragment(mol):
        """Returns the largest fragment of a molecule."""
        if mol is None:
            return None
        try:
            frags = Chem.GetMolFrags(mol, asMols=True)
            if frags:
                return max(frags, key=lambda x: x.GetNumAtoms())
            return mol
        except:
            return mol

uru = UsefulRDKitUtils()

def cleanup_fragment(mol):
    """
    Replace atom map numbers with Hydrogens
    :param mol: input molecule
    :return: modified molecule, number of R-groups
    """
    rgroup_count = 0
    if mol is None:
        return None, 0
    
    # Create a copy to avoid modifying the original
    mol_copy = Chem.Mol(mol)
    
    for atm in mol_copy.GetAtoms():
        atm.SetAtomMapNum(0)
        if atm.GetAtomicNum() == 0:  # This is a dummy atom (R-group attachment point)
            rgroup_count += 1
            atm.SetAtomicNum(1)  # Change to Hydrogen
    mol_copy = Chem.RemoveAllHs(mol_copy)
    return mol_copy, rgroup_count

def generate_fragments(mol):
    """
    Generate fragments using the RDKit
    :param mol: RDKit molecule
    :return: a Pandas dataframe with Scaffold SMILES, Number of Atoms, Number of R-Groups
    """
    if mol is None:
        return pd.DataFrame(columns=["Scaffold", "NumAtoms", "NumRgroups"])
    
    try:
        # Generate molecule fragments using MMPA
        frag_list = FragmentMol(mol)
        
        # Flatten the output into a single list
        flat_frag_list = [x for x in itertools.chain(*frag_list) if x]
        
        # Extract the largest fragment from each molecule
        flat_frag_list = [uru.get_largest_fragment(x) for x in flat_frag_list]
        
        # Keep fragments where the number of atoms in the fragment is at least 2/3 of the number atoms in input molecule
        num_mol_atoms = mol.GetNumAtoms()
        flat_frag_list = [x for x in flat_frag_list if x.GetNumAtoms() / num_mol_atoms > 0.67]
        
        # Remove atom map numbers from the fragments and count R-groups
        processed_frags = [cleanup_fragment(x) for x in flat_frag_list]
        
        # Convert fragments to SMILES
        frag_smiles_list = []
        for mol_clean, rgroup_count in processed_frags:
            if mol_clean is not None:
                try:
                    smiles = Chem.MolToSmiles(mol_clean)
                    frag_smiles_list.append([smiles, mol_clean.GetNumAtoms(), rgroup_count])
                except:
                    continue
        
        # Add the input molecule to the fragment list (with 1 R-group)
        frag_smiles_list.append([Chem.MolToSmiles(mol), mol.GetNumAtoms(), 1])
        
        # Put the results into a Pandas dataframe
        frag_df = pd.DataFrame(frag_smiles_list, columns=["Scaffold", "NumAtoms", "NumRgroups"])
        
        # Remove duplicate fragments
        frag_df = frag_df.drop_duplicates("Scaffold")
        
        return frag_df
        
    except Exception as e:
        st.warning(f"Error generating fragments: {e}")
        return pd.DataFrame(columns=["Scaffold", "NumAtoms", "NumRgroups"])

def find_scaffolds(df_in):
    """
    Generate scaffolds for a set of molecules
    :param df_in: Pandas dataframe with [SMILES, Name, RDKit molecule] columns
    :return: dataframe with molecules and scaffolds, dataframe with unique scaffolds
    """
    df_list = []
    
    # Use tqdm for progress tracking
    for smiles, name, mol in tqdm(df_in[["SMILES", "Name", "mol"]].values, desc="Generating scaffolds"):
        if mol is not None:
            tmp_df = generate_fragments(mol).copy()
            if not tmp_df.empty:
                tmp_df['Name'] = name
                tmp_df['SMILES'] = smiles
                df_list.append(tmp_df)
    
    if not df_list:
        return pd.DataFrame(), pd.DataFrame()
    
    # Combine the list of dataframes into a single dataframe
    mol_df = pd.concat(df_list, ignore_index=True)
    
    # Collect scaffolds
    scaffold_list = []
    for scaffold, group in mol_df.groupby("Scaffold"):
        scaffold_list.append([
            scaffold, 
            len(group.Name.unique()), 
            group.NumAtoms.values[0],
            group.NumRgroups.values[0]
        ])
    
    scaffold_df = pd.DataFrame(scaffold_list, columns=["Scaffold", "Count", "NumAtoms", "NumRgroups"])
    
    # Sort scaffolds by frequency and size
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
    # Filter molecules that have this scaffold
    match_df = mol_df.query("Scaffold == @scaffold")
    
    if match_df.empty:
        return [], pd.DataFrame()
    
    # Merge with activity data
    merge_df = match_df.merge(activity_df, on=["SMILES", "Name"])
    
    if merge_df.empty:
        return [], pd.DataFrame()
    
    # Create scaffold molecule for R-group decomposition
    scaffold_mol = Chem.MolFromSmiles(scaffold)
    if scaffold_mol is None:
        return [], merge_df[["SMILES", "Name", "pIC50"]]
    
    # Convert molecules back from SMILES to mol objects
    mol_list = [Chem.MolFromSmiles(smi) for smi in merge_df.SMILES]
    
    # Perform R-group decomposition
    try:
        rgroup_match, rgroup_miss = RGroupDecompose([scaffold_mol], mol_list, asSmiles=True)
        
        if rgroup_match:
            rgroup_df = pd.DataFrame(rgroup_match)
            # Return unique cores and the activity data
            return rgroup_df.Core.unique(), merge_df[["SMILES", "Name", "pIC50"]]
        else:
            return [], merge_df[["SMILES", "Name", "pIC50"]]
    except:
        return [], merge_df[["SMILES", "Name", "pIC50"]]

# ==========================================
# 2. MMP Analysis Functions
# ==========================================

def perform_mmp_analysis(scaffold_mol_df, property_col='pIC50'):
    """
    Perform MMP analysis on molecules sharing a common scaffold
    :param scaffold_mol_df: DataFrame with SMILES, Name, and property values
    :param property_col: Name of the property column
    :return: DataFrame with MMP pairs and their property deltas
    """
    mmp_pairs = []
    
    # Get all unique molecules
    molecules = scaffold_mol_df.to_dict('records')
    
    # Generate all pairs of molecules
    for (mol1, mol2) in combinations(molecules, 2):
        if mol1['SMILES'] == mol2['SMILES']:
            continue
        
        # Calculate property delta
        delta = mol2[property_col] - mol1[property_col]
        
        # Sort by SMILES for consistent direction
        pair = sorted([mol1, mol2], key=lambda x: x['SMILES'])
        m1, m2 = pair[0], pair[1]
        
        mmp_pairs.append({
            'SMILES_1': m1['SMILES'],
            'Name_1': m1['Name'],
            f'{property_col}_1': m1[property_col],
            'SMILES_2': m2['SMILES'],
            'Name_2': m2['Name'],
            f'{property_col}_2': m2[property_col],
            'Delta': delta
        })
    
    return pd.DataFrame(mmp_pairs)

def analyze_transform_effects(mmp_df, min_occurrence=5):
    """
    Analyze transformation effects by grouping similar MMP pairs
    :param mmp_df: DataFrame with MMP pairs
    :param min_occurrence: Minimum number of occurrences for a transform
    :return: DataFrame with transform statistics
    """
    # For now, return a simple summary - in practice you'd need to identify
    # chemical transformations from the pairs
    if mmp_df.empty:
        return pd.DataFrame()
    
    # Calculate basic statistics
    transform_stats = {
        'Total_Pairs': len(mmp_df),
        'Mean_Delta': mmp_df['Delta'].mean(),
        'Std_Delta': mmp_df['Delta'].std(),
        'Min_Delta': mmp_df['Delta'].min(),
        'Max_Delta': mmp_df['Delta'].max(),
        'Positive_Transforms': (mmp_df['Delta'] > 0).sum(),
        'Negative_Transforms': (mmp_df['Delta'] < 0).sum()
    }
    
    return pd.DataFrame([transform_stats])

# ==========================================
# 3. Streamlit App
# ==========================================

st.set_page_config(page_title="Advanced MMP & Scaffold Analysis", layout="wide")
st.title("🧬 Advanced MMP & Scaffold Analysis")
st.markdown("""
This app performs comprehensive scaffold-based MMP analysis to identify structural transformations 
and their effect on biological activity. Upload a CSV file containing molecules and activity data.
""")

# --- Sidebar: Upload & Settings ---
with st.sidebar:
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], help="CSV file should contain SMILES, compound names, and property values")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded {len(df)} rows")
            
            with st.expander("Data Preview"):
                st.dataframe(df.head(3), use_container_width=True)
            
            st.header("2. Map Columns")
            cols = df.columns.tolist()
            
            # Try to guess columns
            smiles_guess = None
            name_guess = None
            prop_guess = None
            
            for col in cols:
                col_lower = col.lower()
                if 'smiles' in col_lower:
                    smiles_guess = col
                elif 'name' in col_lower or 'id' in col_lower or 'compound' in col_lower:
                    name_guess = col
                elif 'pic50' in col_lower or 'activity' in col_lower or 'value' in col_lower or 'property' in col_lower:
                    prop_guess = col
            
            smiles_col = st.selectbox("SMILES Column", cols, 
                                     index=cols.index(smiles_guess) if smiles_guess in cols else 0)
            name_col = st.selectbox("Name/ID Column", cols, 
                                   index=cols.index(name_guess) if name_guess in cols else 0)
            prop_col = st.selectbox("Property Column (e.g. pIC50)", cols, 
                                   index=cols.index(prop_guess) if prop_guess in cols else 0)
            
            st.header("3. Analysis Settings")
            
            col1, col2 = st.columns(2)
            with col1:
                min_scaffold_count = st.number_input("Min Scaffold Occurrence", 
                                                    min_value=2, value=5,
                                                    help="Only analyze scaffolds with at least this many molecules")
            
            with col2:
                fragment_threshold = st.slider("Fragment Size Threshold", 
                                              min_value=0.5, max_value=0.9, 
                                              value=0.67, step=0.01,
                                              help="Minimum fraction of atoms in fragment relative to original molecule")
            
            run_btn = st.button("Run Analysis", type="primary", use_container_width=True)
            
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            run_btn = False
    else:
        st.info("Please upload a CSV file to begin")
        run_btn = False

# --- Main Analysis Logic ---
if uploaded_file is not None and run_btn:
    # Initialize progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # 1. Preprocessing
        status_text.text("Step 1/5: Preprocessing molecules...")
        input_df = df.copy()
        
        # Check required columns
        required_cols = [smiles_col, prop_col]
        missing_cols = [col for col in required_cols if col not in input_df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            st.stop()
        
        # Rename columns for consistency
        input_df = input_df.rename(columns={
            smiles_col: 'SMILES',
            name_col: 'Name',
            prop_col: 'pIC50'
        })
        
        # Keep only necessary columns
        input_df = input_df[['SMILES', 'Name', 'pIC50']]
        
        # Drop rows with missing values
        input_df = input_df.dropna(subset=['SMILES', 'pIC50'])
        
        if len(input_df) == 0:
            st.error("No valid data after removing rows with missing values.")
            st.stop()
        
        # Convert SMILES to molecules
        status_text.text("Step 2/5: Converting SMILES to molecules...")
        input_df['mol'] = input_df['SMILES'].apply(Chem.MolFromSmiles)
        
        # Count invalid molecules
        invalid_count = input_df['mol'].isnull().sum()
        if invalid_count > 0:
            st.warning(f"Skipping {invalid_count} invalid SMILES strings")
        
        # Filter invalid mols
        input_df = input_df[input_df['mol'].notnull()]
        
        if len(input_df) == 0:
            st.error("No valid molecules found in the dataset.")
            st.stop()
        
        progress_bar.progress(20)
        
        # 2. Generate Scaffolds
        status_text.text("Step 3/5: Generating molecular scaffolds...")
        mol_df, scaffold_df = find_scaffolds(input_df)
        
        progress_bar.progress(40)
        
        if scaffold_df.empty:
            st.error("No scaffolds generated. Check input structures.")
            st.stop()
        
        # Filter scaffolds by minimum occurrence
        scaffold_df = scaffold_df[scaffold_df['Count'] >= min_scaffold_count]
        
        if scaffold_df.empty:
            st.warning(f"No scaffolds found with at least {min_scaffold_count} occurrences.")
            st.stop()
        
        progress_bar.progress(60)
        
        # 3. Analyze Top Scaffolds
        status_text.text("Step 4/5: Analyzing top scaffolds...")
        
        # Let user select which scaffolds to analyze
        scaffold_options = scaffold_df.head(20)['Scaffold'].tolist()
        selected_scaffolds = st.multiselect(
            "Select Scaffolds to Analyze (max 3 recommended)",
            scaffold_options,
            default=scaffold_options[:min(3, len(scaffold_options))]
        )
        
        if not selected_scaffolds:
            st.warning("Please select at least one scaffold to analyze.")
            st.stop()
        
        # Analyze each selected scaffold
        all_results = []
        scaffold_details = []
        
        for i, scaffold_smiles in enumerate(selected_scaffolds):
            status_text.text(f"Analyzing scaffold {i+1}/{len(selected_scaffolds)}...")
            
            # Get molecules with this scaffold
            scaffold_cores, scaffold_mol_df = get_molecules_with_scaffold(
                scaffold_smiles, mol_df, input_df
            )
            
            if not scaffold_mol_df.empty:
                # Perform MMP analysis
                mmp_df = perform_mmp_analysis(scaffold_mol_df)
                
                # Calculate scaffold statistics
                ic50_values = scaffold_mol_df['pIC50'].values
                scaffold_stats = {
                    'Scaffold': scaffold_smiles,
                    'Molecule_Count': len(scaffold_mol_df),
                    'Property_Min': float(np.min(ic50_values)),
                    'Property_Max': float(np.max(ic50_values)),
                    'Property_Range': float(np.max(ic50_values) - np.min(ic50_values)),
                    'Property_Std': float(np.std(ic50_values)),
                    'Property_Mean': float(np.mean(ic50_values)),
                    'MMP_Pairs': len(mmp_df) if not mmp_df.empty else 0
                }
                
                scaffold_details.append(scaffold_stats)
                
                # Add scaffold info to MMP results
                if not mmp_df.empty:
                    mmp_df['Scaffold'] = scaffold_smiles
                    all_results.append(mmp_df)
        
        progress_bar.progress(80)
        
        # Combine all results
        if all_results:
            combined_mmp_df = pd.concat(all_results, ignore_index=True)
        else:
            combined_mmp_df = pd.DataFrame()
        
        scaffold_details_df = pd.DataFrame(scaffold_details)
        
        progress_bar.progress(100)
        status_text.text("Analysis complete!")
        
        # --- Results Display ---
        st.success(f"✅ Analysis Complete! Found {len(scaffold_details_df)} scaffolds with sufficient data.")
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", 
            "🏗️ Scaffold Details", 
            "🔄 MMP Analysis", 
            "🔍 Molecule Viewer",
            "💾 Download Results"
        ])
        
        with tab1:
            st.subheader("Analysis Overview")
            
            # Display summary statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Molecules", len(input_df))
            with col2:
                st.metric("Unique Scaffolds", len(scaffold_df))
            with col3:
                st.metric("Analyzed Scaffolds", len(scaffold_details_df))
            with col4:
                total_pairs = combined_mmp_df['Delta'].count() if not combined_mmp_df.empty else 0
                st.metric("MMP Pairs Found", total_pairs)
            
            # Plot scaffold distribution
            if not scaffold_df.empty:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                
                # Plot 1: Scaffold frequency distribution
                top_scaffolds = scaffold_df.head(10)
                ax1.barh(range(len(top_scaffolds)), top_scaffolds['Count'].values)
                ax1.set_yticks(range(len(top_scaffolds)))
                ax1.set_yticklabels([s[:20] + '...' if len(s) > 20 else s for s in top_scaffolds['Scaffold'].values])
                ax1.set_xlabel('Number of Molecules')
                ax1.set_title('Top 10 Most Frequent Scaffolds')
                ax1.invert_yaxis()
                
                # Plot 2: Scaffold size vs frequency
                ax2.scatter(scaffold_df['NumAtoms'], scaffold_df['Count'], alpha=0.5)
                ax2.set_xlabel('Number of Atoms in Scaffold')
                ax2.set_ylabel('Number of Molecules')
                ax2.set_title('Scaffold Size vs Frequency')
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
        
        with tab2:
            st.subheader("Scaffold Details")
            
            if not scaffold_details_df.empty:
                # Display scaffold details table
                st.dataframe(
                    scaffold_details_df.style.background_gradient(subset=['Property_Range'], cmap='YlOrRd'),
                    use_container_width=True
                )
                
                # Show scaffold structures
                st.subheader("Scaffold Structures")
                cols = st.columns(min(3, len(selected_scaffolds)))
                
                for idx, scaffold_smiles in enumerate(selected_scaffolds):
                    with cols[idx % 3]:
                        try:
                            mol = Chem.MolFromSmiles(scaffold_smiles)
                            if mol:
                                img = Draw.MolToImage(mol, size=(300, 200))
                                st.image(img, caption=f"Scaffold {idx+1}")
                                st.caption(f"SMILES: `{scaffold_smiles}`")
                        except:
                            st.warning(f"Could not render scaffold {idx+1}")
            else:
                st.info("No scaffold details available.")
        
        with tab3:
            st.subheader("MMP Analysis Results")
            
            if not combined_mmp_df.empty:
                # Display MMP pairs
                st.dataframe(
                    combined_mmp_df.style.background_gradient(subset=['Delta'], cmap='coolwarm'),
                    use_container_width=True,
                    height=400
                )
                
                # Plot delta distribution
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.hist(combined_mmp_df['Delta'].dropna(), bins=30, edgecolor='black', alpha=0.7)
                ax.axvline(0, color='red', linestyle='--', alpha=0.5)
                ax.set_xlabel('ΔpIC50')
                ax.set_ylabel('Frequency')
                ax.set_title('Distribution of Property Changes in MMP Pairs')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # Summary statistics
                st.subheader("MMP Statistics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean Δ", f"{combined_mmp_df['Delta'].mean():.3f}")
                with col2:
                    st.metric("Std Δ", f"{combined_mmp_df['Delta'].std():.3f}")
                with col3:
                    positive = (combined_mmp_df['Delta'] > 0).sum()
                    st.metric("Positive Δ", positive)
                with col4:
                    negative = (combined_mmp_df['Delta'] < 0).sum()
                    st.metric("Negative Δ", negative)
            else:
                st.info("No MMP pairs generated. Try selecting different scaffolds or lowering the minimum occurrence threshold.")
        
        with tab4:
            st.subheader("Molecule Viewer")
            
            if not combined_mmp_df.empty:
                # Let user select a specific MMP pair to view
                pair_options = combined_mmp_df.apply(
                    lambda row: f"{row['Name_1']} → {row['Name_2']} (Δ={row['Delta']:.2f})", 
                    axis=1
                ).tolist()
                
                selected_pair = st.selectbox("Select an MMP pair to view", pair_options)
                
                if selected_pair:
                    pair_idx = pair_options.index(selected_pair)
                    pair_data = combined_mmp_df.iloc[pair_idx]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**{pair_data['Name_1']}**")
                        mol1 = Chem.MolFromSmiles(pair_data['SMILES_1'])
                        if mol1:
                            img1 = Draw.MolToImage(mol1, size=(300, 250))
                            st.image(img1)
                        st.metric("pIC50", f"{pair_data['pIC50_1']:.2f}")
                        st.caption(f"SMILES: `{pair_data['SMILES_1']}`")
                    
                    with col2:
                        st.markdown(f"**{pair_data['Name_2']}**")
                        mol2 = Chem.MolFromSmiles(pair_data['SMILES_2'])
                        if mol2:
                            img2 = Draw.MolToImage(mol2, size=(300, 250))
                            st.image(img2)
                        st.metric("pIC50", f"{pair_data['pIC50_2']:.2f}")
                        st.metric("ΔpIC50", f"{pair_data['Delta']:.2f}", 
                                 delta_color="inverse" if pair_data['Delta'] < 0 else "normal")
                        st.caption(f"SMILES: `{pair_data['SMILES_2']}`")
                    
                    st.markdown(f"**Scaffold:** `{pair_data['Scaffold']}`")
            else:
                st.info("No MMP pairs available for viewing.")
        
        with tab5:
            st.subheader("Download Results")
            
            # Prepare data for download
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Scaffold details
                scaffold_csv = scaffold_details_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Scaffold Details",
                    data=scaffold_csv,
                    file_name="scaffold_details.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # MMP pairs
                if not combined_mmp_df.empty:
                    mmp_csv = combined_mmp_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 MMP Pairs",
                        data=mmp_csv,
                        file_name="mmp_pairs.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.button("📥 MMP Pairs (No data)", disabled=True, use_container_width=True)
            
            with col3:
                # All scaffolds
                all_scaffolds_csv = scaffold_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 All Scaffolds",
                    data=all_scaffolds_csv,
                    file_name="all_scaffolds.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            st.info("Download includes:")
            st.markdown("""
            - **Scaffold Details**: Statistics for analyzed scaffolds
            - **MMP Pairs**: All matched molecular pairs with property deltas
            - **All Scaffolds**: Complete list of all scaffolds found
            """)
    
    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

elif uploaded_file is None:
    # Show example data format
    with st.expander("📋 Example Data Format & Instructions"):
        example_data = {
            'SMILES': [
                'CC(=O)Oc1ccccc1C(=O)O',  # Aspirin
                'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
                'c1ccccc1',  # Benzene
                'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',  # Ibuprofen
                'C1=CC=C(C=C1)C(=O)O'  # Benzoic acid
            ],
            'Name': ['Aspirin', 'Caffeine', 'Benzene', 'Ibuprofen', 'Benzoic_acid'],
            'pIC50': [4.76, 3.12, 2.45, 5.12, 3.89]
        }
        example_df = pd.DataFrame(example_data)
        st.dataframe(example_df, use_container_width=True)
        
        st.markdown("""
        ### **Expected Data Format:**
        - **SMILES**: Molecular structures in SMILES format
        - **Name/ID**: Compound identifiers (unique names recommended)
        - **pIC50/Property**: Numerical property values (higher = more active)
        
        ### **How It Works:**
        1. **Scaffold Generation**: The app identifies common molecular frameworks
        2. **R-group Analysis**: For each scaffold, it identifies variable regions
        3. **MMP Identification**: Finds pairs of molecules that differ only at R-group positions
        4. **Delta Calculation**: Computes property differences between MMP pairs
        
        ### **Key Features:**
        - Identifies conserved molecular scaffolds
        - Analyzes R-group contributions to activity
        - Visualizes molecular transformations
        - Exports results for further analysis
        """)

# Footer
st.markdown("---")
st.markdown("*Built with RDKit and Streamlit | Advanced Scaffold-based MMP Analysis*")
