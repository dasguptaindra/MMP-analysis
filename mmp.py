import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem import rdMMPA
from rdkit.Chem.MolStandardize import rdMolStandardize
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# ==========================================
# 1. Helper Functions (Refactored for Standalone Use)
# ==========================================

def get_largest_fragment(mol):
    """
    Returns the largest fragment of a molecule.
    Replaces useful_rdkit_utils.get_largest_fragment
    """
    if mol is None:
        return None
    try:
        lfc = rdMolStandardize.LargestFragmentChooser()
        return lfc.choose(mol)
    except:
        return mol

def remove_map_nums(mol):
    """
    Remove atom map numbers from a molecule (in-place).
    """
    if mol is None:
        return
    for atm in mol.GetAtoms():
        atm.SetAtomMapNum(0)

def sort_fragments(mol_list):
    """
    Sort a list of molecules by number of atoms (largest to smallest).
    """
    frag_num_atoms_list = [(x.GetNumAtoms(), x) for x in mol_list]
    frag_num_atoms_list.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in frag_num_atoms_list]

def fragment_mol_simple(mol):
    """
    Perform single-cut fragmentation using RDKit's MMPA.
    Returns a list of (Core, R_group) pairs sorted by size.
    """
    if mol is None:
        return []
    
    try:
        # maxCuts=1 generates single cuts (producing 2 fragments)
        # rdMMPA.FragmentMol returns a list of tuples: (core, sidechain)
        frags = rdMMPA.FragmentMol(mol, maxCuts=1, resultsAsMols=True)
        
        valid_pairs = []
        for core, sidechain in frags:
            if core is None or sidechain is None:
                continue
                
            # Clone molecules to avoid modifying originals
            m1 = Chem.Mol(core)
            m2 = Chem.Mol(sidechain)
            
            # Sort to identify Core vs R-group (larger = core)
            sorted_frags = sort_fragments([m1, m2])
            core_frag = sorted_frags[0]
            r_group_frag = sorted_frags[1]
            
            # Remove map numbers for canonical SMILES generation
            remove_map_nums(core_frag)
            remove_map_nums(r_group_frag)
            
            valid_pairs.append((core_frag, r_group_frag))
            
        return valid_pairs
    except:
        return []

# ==========================================
# 2. Main Streamlit App
# ==========================================

st.set_page_config(page_title="MMP Analysis App", layout="wide")
st.title("🧪 Matched Molecular Pair (MMP) Analysis")
st.markdown("""
This app performs MMP analysis to identify structural transformations and their effect on a property (e.g., pIC50).
Upload a CSV file containing molecules and activity data.
""")

# --- Sidebar: Upload & Settings ---
with st.sidebar:
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], help="CSV file should contain SMILES, compound names, and property values")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded {len(df)} rows")
            st.write("Data Preview:")
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
                elif 'pic50' in col_lower or 'pactivity' in col_lower or 'value' in col_lower or 'property' in col_lower:
                    prop_guess = col
            
            smiles_col = st.selectbox("SMILES Column", cols, index=cols.index(smiles_guess) if smiles_guess in cols else 0)
            name_col = st.selectbox("Name/ID Column", cols, index=cols.index(name_guess) if name_guess in cols else 0)
            prop_col = st.selectbox("Property Column (e.g. pIC50)", cols, index=cols.index(prop_guess) if prop_guess in cols else 0)
            
            st.header("3. Analysis Settings")
            min_occurence = st.number_input("Min Transform Occurrence", min_value=2, value=5, 
                                          help="Only show transformations that occur at least this many times")
            
            run_btn = st.button("Run Analysis", type="primary", use_container_width=True)
            
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            run_btn = False
    else:
        st.info("Please upload a CSV file to begin")
        run_btn = False

# --- Main Analysis Logic ---
if 'uploaded_file' in locals() and uploaded_file is not None and run_btn:
    st.info("Running MMP Analysis... This may take a moment.")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # 1. Preprocessing
        status_text.text("Step 1/4: Preprocessing molecules...")
        input_df = df.copy()
        
        # Check if required columns exist
        required_cols = [smiles_col, prop_col]
        missing_cols = [col for col in required_cols if col not in input_df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            st.stop()
        
        # Drop rows with missing values
        input_df = input_df.dropna(subset=[smiles_col, prop_col])
        
        if len(input_df) == 0:
            st.error("No valid data after removing rows with missing values.")
            st.stop()
        
        # Add Mol column
        status_text.text("Step 2/4: Converting SMILES to molecules...")
        input_df['mol'] = input_df[smiles_col].apply(Chem.MolFromSmiles)
        
        # Count invalid molecules
        invalid_count = input_df['mol'].isnull().sum()
        if invalid_count > 0:
            st.warning(f"Skipping {invalid_count} invalid SMILES strings")
        
        # Filter invalid mols
        input_df = input_df[input_df['mol'].notnull()]
        
        if len(input_df) == 0:
            st.error("No valid molecules found in the dataset.")
            st.stop()
        
        # Get largest fragment
        input_df['mol'] = input_df['mol'].apply(get_largest_fragment)
        
        progress_bar.progress(25)
        
        # 2. Fragmentation Loop
        status_text.text("Step 3/4: Fragmenting molecules...")
        row_list = []
        
        # Iterate over rows
        total_rows = len(input_df)
        for idx, row in input_df.iterrows():
            mol = row['mol']
            name = row[name_col] if name_col in row else f"Compound_{idx}"
            val = row[prop_col]
            smi = row[smiles_col]
            
            # Fragment
            pairs = fragment_mol_simple(mol)
            
            for core, r_group in pairs:
                # Generate SMILES
                try:
                    core_smi = Chem.MolToSmiles(core)
                    r_smi = Chem.MolToSmiles(r_group)
                    
                    row_list.append([smi, core_smi, r_smi, name, val])
                except:
                    continue
        
        row_df = pd.DataFrame(row_list, columns=["SMILES", "Core", "R_group", "Name", "Property"])
        
        progress_bar.progress(50)
        
        if len(row_df) == 0:
            st.error("No fragments generated. Check input structures.")
            st.stop()
        
        # 3. Pair Generation (Delta calculation)
        status_text.text("Step 4/4: Generating molecular pairs...")
        delta_list = []
        
        # Group by Core to find molecules sharing the same scaffold
        grouped = row_df.groupby("Core")
        
        for core_smi, group in grouped:
            if len(group) > 1:
                # Generate combinations of 2
                group_recs = group.to_dict('records')
                
                for r_a, r_b in combinations(group_recs, 2):
                    if r_a['SMILES'] == r_b['SMILES']:
                        continue
                    
                    # Sort pair by SMILES to maintain consistent direction
                    pair = sorted([r_a, r_b], key=lambda x: x['SMILES'])
                    reagent_a, reagent_b = pair[0], pair[1]
                    
                    delta = reagent_b['Property'] - reagent_a['Property']
                    
                    # Create transform string
                    trans_str = f"{reagent_a['R_group'].replace('*','*-')}>>{reagent_b['R_group'].replace('*','*-')}"
                    
                    # Store result
                    delta_list.append([
                        reagent_a['SMILES'], reagent_a['Core'], reagent_a['R_group'], 
                        reagent_a['Name'], reagent_a['Property'],
                        reagent_b['SMILES'], reagent_b['Core'], reagent_b['R_group'], 
                        reagent_b['Name'], reagent_b['Property'],
                        trans_str, delta
                    ])
        
        progress_bar.progress(75)
        
        delta_cols = [
            "SMILES_1", "Core_1", "R_group_1", "Name_1", "Property_1",
            "SMILES_2", "Core_2", "R_group_2", "Name_2", "Property_2",
            "Transform", "Delta"
        ]
        delta_df = pd.DataFrame(delta_list, columns=delta_cols)
        
        # 4. Aggregation (MMP Table)
        status_text.text("Finalizing results...")
        mmp_list = []
        
        if not delta_df.empty:
            for trans, group in delta_df.groupby("Transform"):
                if len(group) >= min_occurence:
                    mmp_list.append([trans, len(group), group['Delta'].values, group['Delta'].mean()])
        
        mmp_df = pd.DataFrame(mmp_list, columns=["Transform", "Count", "Deltas", "Mean_Delta"])
        
        if not mmp_df.empty:
            mmp_df = mmp_df.sort_values("Mean_Delta", ascending=False)
            mmp_df = mmp_df.reset_index(drop=True)
        
        progress_bar.progress(100)
        status_text.text("Analysis complete!")
        
        # --- Results Display ---
        
        if not mmp_df.empty:
            st.success(f"✅ Analysis Complete! Found {len(mmp_df)} distinct transforms occurring ≥ {min_occurence} times.")
            
            # Tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview & Plots", "📋 MMP Table", "🔍 Detailed Pairs", "💾 Download Results"])
            
            with tab1:
                st.subheader("Distribution of Effects")
                
                # Plot the top transforms
                top_n = st.slider("Show Top N Transforms", 5, 50, 20, key="top_n_slider")
                
                # Prepare data for plotting
                plot_data = []
                subset_df = mmp_df.head(top_n) if len(mmp_df) > top_n else mmp_df
                
                for idx, row in subset_df.iterrows():
                    for d in row['Deltas']:
                        plot_data.append({'Transform': row['Transform'], 'Delta': d})
                
                plot_df = pd.DataFrame(plot_data)
                
                if not plot_df.empty:
                    # Create figure with subplots
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    # Box plot
                    sns.boxplot(data=plot_df, x='Delta', y='Transform', ax=ax1, orient='h')
                    ax1.axvline(0, color='red', linestyle='--', alpha=0.5)
                    ax1.set_title("Property Delta Distribution by Transform")
                    ax1.set_xlabel("ΔProperty")
                    
                    # Violin plot for distribution
                    if len(subset_df) > 1:
                        sns.violinplot(data=plot_df, x='Delta', y='Transform', ax=ax2, orient='h')
                        ax2.axvline(0, color='red', linestyle='--', alpha=0.5)
                        ax2.set_title("Delta Distribution (Violin Plot)")
                        ax2.set_xlabel("ΔProperty")
                    else:
                        ax2.text(0.5, 0.5, "Need at least 2 transforms\nfor violin plot", 
                                ha='center', va='center', transform=ax2.transAxes)
                        ax2.set_title("Violin Plot")
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Summary statistics
                    st.subheader("Summary Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Pairs", len(delta_df))
                    with col2:
                        st.metric("Unique Transforms", len(mmp_df))
                    with col3:
                        st.metric("Avg Delta (All)", f"{delta_df['Delta'].mean():.3f}")
                    with col4:
                        st.metric("Max Delta", f"{delta_df['Delta'].max():.3f}")
                else:
                    st.warning("No data to plot")
                    
            with tab2:
                st.subheader("MMP Summary Table")
                
                # Add formatted columns
                display_df = mmp_df.copy()
                display_df['Mean_Delta'] = display_df['Mean_Delta'].round(3)
                display_df['Std_Delta'] = display_df['Deltas'].apply(lambda x: np.std(x) if len(x) > 1 else 0).round(3)
                display_df['Min_Delta'] = display_df['Deltas'].apply(lambda x: min(x)).round(3)
                display_df['Max_Delta'] = display_df['Deltas'].apply(lambda x: max(x)).round(3)
                
                st.dataframe(
                    display_df[['Transform', 'Count', 'Mean_Delta', 'Std_Delta', 'Min_Delta', 'Max_Delta']]
                    .style.background_gradient(subset=['Mean_Delta'], cmap='coolwarm'),
                    use_container_width=True,
                    height=400
                )
                
            with tab3:
                st.subheader("Drill Down into Specific Transforms")
                
                if not mmp_df.empty:
                    selected_trans = st.selectbox("Select a Transform to View Pairs", 
                                                 mmp_df['Transform'].tolist(),
                                                 key="transform_select")
                    
                    if selected_trans:
                        subset = delta_df[delta_df['Transform'] == selected_trans]
                        st.write(f"Showing {len(subset)} pairs for: **{selected_trans}**")
                        
                        # Display statistics for this transform
                        trans_stats = mmp_df[mmp_df['Transform'] == selected_trans].iloc[0]
                        cols = st.columns(4)
                        cols[0].metric("Mean Δ", f"{trans_stats['Mean_Delta']:.3f}")
                        cols[1].metric("Count", trans_stats['Count'])
                        cols[2].metric("Std Dev", f"{np.std(trans_stats['Deltas']):.3f}")
                        cols[3].metric("Range", f"{min(trans_stats['Deltas']):.3f} to {max(trans_stats['Deltas']):.3f}")
                        
                        # Show each pair
                        st.subheader("Individual Pairs")
                        for i, row in subset.iterrows():
                            with st.expander(f"Pair {i+1}: {row['Name_1']} → {row['Name_2']} (Δ={row['Delta']:.3f})"):
                                c1, c2 = st.columns(2)
                                
                                with c1:
                                    st.markdown(f"**{row['Name_1']}**")
                                    mol1 = Chem.MolFromSmiles(row['SMILES_1'])
                                    if mol1:
                                        img1 = Draw.MolToImage(mol1, size=(300, 200))
                                        st.image(img1)
                                    st.markdown(f"**Property:** {row['Property_1']:.3f}")
                                    st.markdown(f"**R-group:** `{row['R_group_1']}`")
                                        
                                with c2:
                                    st.markdown(f"**{row['Name_2']}**")
                                    mol2 = Chem.MolFromSmiles(row['SMILES_2'])
                                    if mol2:
                                        img2 = Draw.MolToImage(mol2, size=(300, 200))
                                        st.image(img2)
                                    st.markdown(f"**Property:** {row['Property_2']:.3f}")
                                    st.markdown(f"**R-group:** `{row['R_group_2']}`")
                                
                                st.markdown(f"**Transform:** `{row['Transform']}`")
                                st.markdown(f"**ΔProperty:** `{row['Delta']:.3f}`")
                else:
                    st.info("No transforms available for drill-down")
                    
            with tab4:
                st.subheader("Download Results")
                
                # Prepare data for download
                # 1. MMP summary
                summary_csv = mmp_df[['Transform', 'Count', 'Mean_Delta']].to_csv(index=False)
                summary_csv = summary_csv.encode('utf-8')
                
                # 2. All pairs
                pairs_csv = delta_df.to_csv(index=False)
                pairs_csv = pairs_csv.encode('utf-8')
                
                # 3. Raw fragments
                fragments_csv = row_df.to_csv(index=False)
                fragments_csv = fragments_csv.encode('utf-8')
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.download_button(
                        label="📥 Download MMP Summary",
                        data=summary_csv,
                        file_name="mmp_summary.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    st.download_button(
                        label="📥 Download All Pairs",
                        data=pairs_csv,
                        file_name="all_mmp_pairs.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col3:
                    st.download_button(
                        label="📥 Download Fragments",
                        data=fragments_csv,
                        file_name="fragments.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                st.info("Files include:")
                st.markdown("""
                - **MMP Summary**: Unique transformations with counts and mean Δ
                - **All Pairs**: Every matched molecular pair with details
                - **Fragments**: All core-Rgroup fragments from initial fragmentation
                """)
        
        else:
            st.warning(f"No transforms found with occurrence ≥ {min_occurence}. Try lowering the threshold.")
            
            # Show delta_df statistics if no mmp_df
            if not delta_df.empty:
                st.subheader("Delta Distribution Statistics")
                st.write(f"Total pairs generated: {len(delta_df)}")
                st.write(f"Unique transforms: {delta_df['Transform'].nunique()}")
                
                # Show histogram of delta values
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.hist(delta_df['Delta'], bins=30, edgecolor='black', alpha=0.7)
                ax.axvline(delta_df['Delta'].mean(), color='red', linestyle='--', label=f"Mean: {delta_df['Delta'].mean():.3f}")
                ax.set_xlabel("ΔProperty")
                ax.set_ylabel("Frequency")
                ax.set_title("Distribution of All Property Deltas")
                ax.legend()
                st.pyplot(fig)
    
    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")
        st.exception(e)

elif uploaded_file is None:
    # Show example data format
    with st.expander("📋 Example Data Format"):
        example_data = {
            'SMILES': ['CC(=O)Oc1ccccc1C(=O)O', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C', 'c1ccccc1'],
            'Name': ['Aspirin', 'Caffeine', 'Benzene'],
            'pIC50': [4.76, 3.12, 2.45]
        }
        example_df = pd.DataFrame(example_data)
        st.dataframe(example_df, use_container_width=True)
        
        st.markdown("""
        **Expected columns:**
        - **SMILES**: Molecular structures in SMILES format
        - **Name/ID**: Compound identifiers (optional but recommended)
        - **Property**: Numerical property values (e.g., pIC50, activity)
        
        **Note:** The app will work with any column names - you can map them in the sidebar.
        """)

# Footer
st.markdown("---")
st.markdown("*Built with RDKit and Streamlit*")
