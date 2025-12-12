import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.rdMMPA import FragmentMol
from rdkit.Chem.rdRGroupDecomposition import RGroupDecompose
import itertools
import matplotlib.pyplot as plt
from itertools import combinations

# ==========================================
# Enhanced Helper Functions
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

def generate_fragments_enhanced(mol, min_fragment_size=0.5):
    """
    Enhanced fragment generation with better error handling
    :param mol: RDKit molecule
    :param min_fragment_size: Minimum fraction of atoms to keep
    :return: a Pandas dataframe with Scaffold SMILES, Number of Atoms, Number of R-Groups
    """
    if mol is None:
        return pd.DataFrame(columns=["Scaffold", "NumAtoms", "NumRgroups"])
    
    try:
        # Try to sanitize molecule first
        Chem.SanitizeMol(mol)
        
        # Generate molecule fragments using MMPA with try-catch
        frag_list = []
        try:
            frag_list = FragmentMol(mol, maxCuts=2, resultsAsMols=False, pattern="[#6+0;!$(*=,#[!#6])]!@!=!#[*]")
        except Exception as e:
            # Fallback: return the molecule itself as a scaffold
            frag_smiles = Chem.MolToSmiles(mol)
            frag_df = pd.DataFrame([[frag_smiles, mol.GetNumAtoms(), 1]], 
                                 columns=["Scaffold", "NumAtoms", "NumRgroups"])
            return frag_df
        
        # If no fragments generated, use molecule itself
        if not frag_list:
            frag_smiles = Chem.MolToSmiles(mol)
            frag_df = pd.DataFrame([[frag_smiles, mol.GetNumAtoms(), 1]], 
                                 columns=["Scaffold", "NumAtoms", "NumRgroups"])
            return frag_df
        
        # Flatten the output into a single list
        flat_frag_list = [Chem.MolFromSmiles(x[0]) for x in frag_list]
        flat_frag_list = [x for x in flat_frag_list if x is not None]
        
        # Extract the largest fragment from each molecule
        flat_frag_list = [uru.get_largest_fragment(x) for x in flat_frag_list]
        flat_frag_list = [x for x in flat_frag_list if x is not None]
        
        # Keep fragments where the number of atoms in the fragment meets minimum size
        num_mol_atoms = mol.GetNumAtoms()
        if num_mol_atoms > 0:
            flat_frag_list = [x for x in flat_frag_list if x.GetNumAtoms() / num_mol_atoms >= min_fragment_size]
        
        # If no fragments after filtering, use the molecule itself
        if not flat_frag_list:
            frag_smiles = Chem.MolToSmiles(mol)
            frag_df = pd.DataFrame([[frag_smiles, mol.GetNumAtoms(), 1]], 
                                 columns=["Scaffold", "NumAtoms", "NumRgroups"])
            return frag_df
        
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
        # Return the molecule itself as a fallback
        try:
            frag_smiles = Chem.MolToSmiles(mol)
            return pd.DataFrame([[frag_smiles, mol.GetNumAtoms(), 1]], 
                              columns=["Scaffold", "NumAtoms", "NumRgroups"])
        except:
            return pd.DataFrame(columns=["Scaffold", "NumAtoms", "NumRgroups"])

def find_scaffolds_enhanced(df_in, min_fragment_size=0.5):
    """
    Enhanced scaffold generation with better error handling
    :param df_in: Pandas dataframe with [SMILES, Name, RDKit molecule] columns
    :param min_fragment_size: Minimum fragment size relative to original molecule
    :return: dataframe with molecules and scaffolds, dataframe with unique scaffolds
    """
    df_list = []
    processed_molecules = 0
    total_molecules = len(df_in)
    
    # Simple progress indicator
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    for idx, (smiles, name, mol) in enumerate(df_in[["SMILES", "Name", "mol"]].values):
        progress_text.text(f"Processing molecule {idx+1}/{total_molecules}: {name}")
        progress_bar.progress((idx + 1) / total_molecules)
        
        if mol is not None:
            try:
                tmp_df = generate_fragments_enhanced(mol, min_fragment_size).copy()
                if not tmp_df.empty:
                    tmp_df['Name'] = name
                    tmp_df['SMILES'] = smiles
                    # Add pIC50 if available in original df
                    if 'pIC50' in df_in.columns:
                        tmp_df['pIC50'] = df_in[df_in['SMILES'] == smiles]['pIC50'].iloc[0] if len(df_in[df_in['SMILES'] == smiles]) > 0 else 0
                    df_list.append(tmp_df)
                    processed_molecules += 1
            except Exception as e:
                continue
    
    progress_text.empty()
    progress_bar.empty()
    st.info(f"Successfully processed {processed_molecules}/{total_molecules} molecules")
    
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
            group.NumRgroups.values[0],
            group.pIC50.mean() if 'pIC50' in group.columns else 0
        ])
    
    scaffold_df = pd.DataFrame(scaffold_list, columns=["Scaffold", "Count", "NumAtoms", "NumRgroups", "Avg_pIC50"])
    
    # Sort scaffolds by frequency and size
    scaffold_df = scaffold_df.sort_values(["Count", "NumAtoms"], ascending=[False, False])
    
    return mol_df, scaffold_df

def simple_scaffold_analysis(mol):
    """
    Simple scaffold generation as fallback
    :param mol: RDKit molecule
    :return: Murcko scaffold SMILES
    """
    if mol is None:
        return ""
    try:
        # Get Murcko scaffold
        scaffold = Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol(mol)
        scaffold = Chem.RemoveHs(scaffold)
        return Chem.MolToSmiles(scaffold)
    except:
        try:
            # Fallback: use the molecule itself
            return Chem.MolToSmiles(mol)
        except:
            return ""

# ==========================================
# Streamlit App
# ==========================================

st.set_page_config(page_title="Advanced Scaffold Analysis", layout="wide")
st.title("🧬 Advanced Scaffold Analysis")
st.markdown("""
This app performs comprehensive scaffold analysis to identify common molecular frameworks 
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
                                                    min_value=1, value=2,
                                                    help="Only analyze scaffolds with at least this many molecules")
            
            with col2:
                fragment_threshold = st.slider("Fragment Size Threshold", 
                                              min_value=0.2, max_value=0.9, 
                                              value=0.5, step=0.05,
                                              help="Minimum fraction of atoms in fragment relative to original molecule")
            
            # Analysis method selection
            analysis_method = st.radio(
                "Scaffold Generation Method",
                ["Enhanced MMPA (Recommended)", "Simple Murcko Scaffolds"],
                help="MMPA is more sophisticated but may fail on some molecules. Murcko is simpler but more robust."
            )
            
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
    main_progress_bar = st.progress(0)
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
        main_progress_bar.progress(20)
        
        # Add error handling for SMILES conversion
        input_df['mol'] = input_df['SMILES'].apply(lambda x: Chem.MolFromSmiles(str(x)) if pd.notna(x) else None)
        
        # Count invalid molecules
        invalid_count = input_df['mol'].isnull().sum()
        if invalid_count > 0:
            st.warning(f"Skipping {invalid_count} invalid SMILES strings")
        
        # Filter invalid mols
        input_df = input_df[input_df['mol'].notnull()]
        
        if len(input_df) == 0:
            st.error("No valid molecules found in the dataset.")
            st.stop()
        
        st.info(f"Processing {len(input_df)} valid molecules...")
        main_progress_bar.progress(40)
        
        # 2. Generate Scaffolds
        status_text.text("Step 3/5: Generating molecular scaffolds...")
        
        if analysis_method == "Simple Murcko Scaffolds":
            # Use simple Murcko scaffold analysis
            input_df['Scaffold'] = input_df['mol'].apply(simple_scaffold_analysis)
            
            # Create scaffold dataframe
            scaffold_counts = input_df.groupby('Scaffold').size().reset_index(name='Count')
            scaffold_atoms = input_df.groupby('Scaffold')['mol'].apply(
                lambda x: x.iloc[0].GetNumAtoms() if len(x) > 0 else 0
            ).reset_index(name='NumAtoms')
            
            scaffold_df = pd.merge(scaffold_counts, scaffold_atoms, on='Scaffold')
            scaffold_df['NumRgroups'] = 1  # Default for simple scaffolds
            scaffold_df['Avg_pIC50'] = input_df.groupby('Scaffold')['pIC50'].mean().values
            
            # Sort by frequency
            scaffold_df = scaffold_df.sort_values(['Count', 'NumAtoms'], ascending=[False, False])
            
            # For this simple method, mol_df is just the input with scaffolds
            mol_df = input_df.copy()
            
        else:
            # Use enhanced MMPA method
            mol_df, scaffold_df = find_scaffolds_enhanced(input_df, fragment_threshold)
        
        main_progress_bar.progress(60)
        
        if scaffold_df.empty:
            st.error("No scaffolds generated. Check input structures.")
            
            # Try simple scaffold method as fallback
            st.info("Attempting simple scaffold generation as fallback...")
            input_df['Scaffold'] = input_df['mol'].apply(simple_scaffold_analysis)
            
            scaffold_counts = input_df.groupby('Scaffold').size().reset_index(name='Count')
            if scaffold_counts.empty:
                st.stop()
            
            scaffold_atoms = input_df.groupby('Scaffold')['mol'].apply(
                lambda x: x.iloc[0].GetNumAtoms() if len(x) > 0 else 0
            ).reset_index(name='NumAtoms')
            
            scaffold_df = pd.merge(scaffold_counts, scaffold_atoms, on='Scaffold')
            scaffold_df['NumRgroups'] = 1
            scaffold_df['Avg_pIC50'] = input_df.groupby('Scaffold')['pIC50'].mean().values
            scaffold_df = scaffold_df.sort_values(['Count', 'NumAtoms'], ascending=[False, False])
            mol_df = input_df.copy()
        
        # Filter scaffolds by minimum occurrence
        scaffold_df = scaffold_df[scaffold_df['Count'] >= min_scaffold_count]
        
        if scaffold_df.empty:
            st.warning(f"No scaffolds found with at least {min_scaffold_count} occurrences.")
            # Show what was found
            if 'Scaffold' in input_df.columns:
                all_scaffolds = input_df['Scaffold'].value_counts()
                st.write("All scaffolds found:")
                st.dataframe(all_scaffolds.reset_index().rename(columns={'count': 'Frequency'}))
            st.stop()
        
        main_progress_bar.progress(80)
        
        # 3. Analyze Top Scaffolds
        status_text.text("Step 4/5: Analyzing top scaffolds...")
        
        # Let user select which scaffolds to analyze
        scaffold_options = scaffold_df.head(20)['Scaffold'].tolist()
        selected_scaffolds = st.multiselect(
            "Select Scaffolds to Analyze (max 5 recommended)",
            scaffold_options,
            default=scaffold_options[:min(3, len(scaffold_options))]
        )
        
        if not selected_scaffolds:
            st.warning("Please select at least one scaffold to analyze.")
            st.stop()
        
        # Ensure we have the 'Scaffold' column in input_df for visualization
        if 'Scaffold' not in input_df.columns:
            if analysis_method == "Simple Murcko Scaffolds":
                # Already added above
                pass
            else:
                # For MMPA method, need to merge scaffold info back to input_df
                scaffold_map = mol_df[['SMILES', 'Scaffold']].drop_duplicates()
                input_df = input_df.merge(scaffold_map, on='SMILES', how='left')
        
        main_progress_bar.progress(100)
        status_text.text("Step 5/5: Preparing results...")
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Overview", 
            "🏗️ Scaffold Details", 
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
                st.metric("Selected Scaffolds", len(selected_scaffolds))
            with col4:
                st.metric("Avg Molecules per Scaffold", f"{scaffold_df['Count'].mean():.1f}")
            
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
                
                # Plot 2: Property distribution by scaffold - only if 'Scaffold' exists in input_df
                if 'Scaffold' in input_df.columns:
                    for scaffold in selected_scaffolds[:5]:  # Limit to 5 for clarity
                        scaffold_data = input_df[input_df['Scaffold'] == scaffold]['pIC50']
                        if len(scaffold_data) > 0:
                            ax2.hist(scaffold_data, alpha=0.5, label=scaffold[:15] + '...', bins=10)
                    
                    ax2.set_xlabel('pIC50')
                    ax2.set_ylabel('Frequency')
                    ax2.set_title('Property Distribution by Scaffold')
                    if selected_scaffolds:  # Only add legend if we have scaffolds
                        ax2.legend()
                    ax2.grid(True, alpha=0.3)
                else:
                    ax2.text(0.5, 0.5, 'Scaffold data not available for visualization', 
                            ha='center', va='center', transform=ax2.transAxes)
                    ax2.set_title('Property Distribution by Scaffold')
                
                plt.tight_layout()
                st.pyplot(fig)
        
        with tab2:
            st.subheader("Scaffold Details")
            
            # Display scaffold details table
            st.dataframe(
                scaffold_df.head(20).style.background_gradient(subset=['Count'], cmap='YlOrRd'),
                use_container_width=True
            )
            
            # Show scaffold structures
            if selected_scaffolds:
                st.subheader("Selected Scaffold Structures")
                cols = st.columns(min(3, len(selected_scaffolds)))
                
                for idx, scaffold_smiles in enumerate(selected_scaffolds):
                    with cols[idx % 3]:
                        try:
                            mol = Chem.MolFromSmiles(scaffold_smiles)
                            if mol:
                                img = Draw.MolToImage(mol, size=(300, 200))
                                st.image(img, caption=f"Scaffold {idx+1}")
                                st.caption(f"SMILES: `{scaffold_smiles}`")
                                count = scaffold_df[scaffold_df['Scaffold'] == scaffold_smiles]['Count'].iloc[0] if not scaffold_df.empty else 0
                                st.caption(f"Molecules: {count}")
                        except:
                            st.warning(f"Could not render scaffold {idx+1}")
            else:
                st.info("No scaffolds selected for display.")
        
        with tab3:
            st.subheader("Molecule Viewer")
            
            if 'Scaffold' in input_df.columns and selected_scaffolds:
                # Select scaffold to view molecules
                selected_scaffold = st.selectbox("Select a scaffold to view molecules", selected_scaffolds)
                
                if selected_scaffold:
                    # Get molecules with this scaffold
                    scaffold_molecules = input_df[input_df['Scaffold'] == selected_scaffold]
                    
                    st.write(f"Found {len(scaffold_molecules)} molecules with this scaffold")
                    
                    # Display molecules in a grid
                    st.dataframe(
                        scaffold_molecules[['Name', 'SMILES', 'pIC50']].sort_values('pIC50', ascending=False),
                        use_container_width=True,
                        height=300
                    )
                    
                    # Show some example molecules
                    st.subheader("Example Molecules")
                    example_mols = scaffold_molecules.head(4)
                    
                    cols = st.columns(4)
                    for idx, (_, row) in enumerate(example_mols.iterrows()):
                        with cols[idx % 4]:
                            try:
                                mol = Chem.MolFromSmiles(row['SMILES'])
                                if mol:
                                    img = Draw.MolToImage(mol, size=(200, 150))
                                    st.image(img, caption=row['Name'])
                                    st.caption(f"pIC50: {row['pIC50']:.2f}")
                            except:
                                st.warning(f"Could not render {row['Name']}")
            else:
                st.info("No scaffold data available for molecule viewing.")
        
        with tab4:
            st.subheader("Download Results")
            
            # Prepare data for download
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Scaffold details
                scaffold_csv = scaffold_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 All Scaffolds",
                    data=scaffold_csv,
                    file_name="scaffold_details.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Molecule data with scaffolds
                molecule_data = input_df.copy()
                if 'mol' in molecule_data.columns:
                    molecule_data = molecule_data.drop('mol', axis=1)
                molecule_csv = molecule_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Molecules with Scaffolds",
                    data=molecule_csv,
                    file_name="molecules_with_scaffolds.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col3:
                # Analysis report
                report_data = {
                    'Analysis_Date': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')],
                    'Total_Molecules': [len(input_df)],
                    'Unique_Scaffolds': [len(scaffold_df)],
                    'Avg_Molecules_per_Scaffold': [scaffold_df['Count'].mean()],
                    'Method_Used': [analysis_method]
                }
                report_df = pd.DataFrame(report_data)
                report_csv = report_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Analysis Report",
                    data=report_csv,
                    file_name="analysis_report.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        status_text.empty()
        main_progress_bar.empty()
        st.success("✅ Analysis Complete!")
    
    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

elif uploaded_file is None:
    # Show example data format
    with st.expander("📋 Example Data Format & Instructions"):
        # Create a more diverse example dataset
        example_data = {
            'SMILES': [
                'CC(=O)Oc1ccccc1C(=O)O',  # Aspirin
                'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
                'c1ccccc1',  # Benzene
                'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',  # Ibuprofen
                'C1=CC=C(C=C1)C(=O)O',  # Benzoic acid
                'COc1ccc2cc(C(C)C)ccc2c1',  # Simple aromatic
                'CCCCCCC(=O)O',  # Heptanoic acid
                'CC(C)Cc1ccc(cc1)C(C)C',  # Biphenyl derivative
                'CCOC(=O)c1ccccc1',  # Ethyl benzoate
                'NC(=O)c1ccccc1'  # Benzamide
            ],
            'Name': ['Aspirin', 'Caffeine', 'Benzene', 'Ibuprofen', 'Benzoic_acid', 
                    'Compound_6', 'Heptanoic_acid', 'Biphenyl_deriv', 'Ethyl_benzoate', 'Benzamide'],
            'pIC50': [4.76, 3.12, 2.45, 5.12, 3.89, 4.21, 3.45, 4.67, 3.98, 4.33]
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
        2. **Scaffold Analysis**: Groups molecules by their core scaffolds
        3. **Property Analysis**: Analyzes activity distribution within scaffold families
        4. **Visualization**: Displays scaffold structures and molecule examples
        
        ### **Tips for Success:**
        1. **Start with Simple Murcko Scaffolds** for more robust results
        2. **Lower thresholds** if you have a small dataset
        3. **Check your SMILES** for validity
        4. **Use the example data** to test the app first
        """)

# Footer
st.markdown("---")
st.markdown("*Built with RDKit and Streamlit | Advanced Scaffold Analysis Tool*")
