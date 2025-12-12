import streamlit as st
import pandas as pd
import requests
import os
import sys
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from operator import itemgetter
from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt
import mols2grid
import streamlit.components.v1 as components
import io

# ---------------------------------------------------------
# 1. SETUP & DEPENDENCIES
# ---------------------------------------------------------

st.set_page_config(page_title="MMP Analysis App", layout="wide")

@st.cache_resource
def setup_dependencies():
    """Downloads scaffold_finder.py if not present, replicating the notebook setup."""
    if not os.path.exists("scaffold_finder.py"):
        url = "https://raw.githubusercontent.com/PatWalters/practical_cheminformatics_tutorials/main/sar_analysis/scaffold_finder.py"
        try:
            response = requests.get(url)
            with open("scaffold_finder.py", "w") as f:
                f.write(response.text)
        except Exception as e:
            st.error(f"Failed to download scaffold_finder.py: {e}")
            return False
    return True

# Ensure dependencies are ready
if setup_dependencies():
    try:
        from scaffold_finder import FragmentMol
    except ImportError:
        st.error("Could not import FragmentMol from scaffold_finder.py. Please check the file.")
        st.stop()

# ---------------------------------------------------------
# 2. HELPER FUNCTIONS
# ---------------------------------------------------------

def get_largest_fragment(mol):
    """Replicates useful_rdkit_utils.get_largest_fragment"""
    frags = Chem.GetMolFrags(mol, asMols=True)
    if not frags: return mol
    return max(frags, key=lambda x: x.GetNumAtoms())

def remove_map_nums(mol):
    for atm in mol.GetAtoms():
        atm.SetAtomMapNum(0)

def sort_fragments(mol):
    frag_list = list(Chem.GetMolFrags(mol, asMols=True))
    [remove_map_nums(x) for x in frag_list]
    frag_num_atoms_list = [(x.GetNumAtoms(), x) for x in frag_list]
    frag_num_atoms_list.sort(key=itemgetter(0), reverse=True)
    return [x[1] for x in frag_num_atoms_list]

def rxn_to_image(rxn_mol):
    """Helper to convert reaction to image for display"""
    return Draw.ReactionToImage(rxn_mol, subImgSize=(300, 150))

# ---------------------------------------------------------
# 3. MAIN DATA PROCESSING PIPELINE
# ---------------------------------------------------------

@st.cache_data
def process_data(df, smiles_col, name_col, act_col):
    """Step 1: Preprocessing & Fragmentation"""
    
    # Preprocessing
    df['mol'] = df[smiles_col].apply(Chem.MolFromSmiles)
    # Remove rows where mol creation failed
    df = df.dropna(subset=['mol'])
    df['mol'] = df['mol'].apply(get_largest_fragment)
    
    # Decompose to Scaffolds
    row_list = []
    
    # Using a progress bar for the loop
    progress_bar = st.progress(0)
    total_rows = len(df)
    
    for i, (idx, row) in enumerate(df.iterrows()):
        smiles = row[smiles_col]
        name = row[name_col]
        pIC50 = row[act_col]
        mol = row['mol']
        
        frag_list = FragmentMol(mol, maxCuts=1)
        
        for _, frag_mol in frag_list:
            pair_list = sort_fragments(frag_mol)
            # Ensure we have a core and an R-group
            if len(pair_list) >= 2:
                # Assuming first is larger (core) due to sort? 
                # The notebook takes all parts. Let's follow notebook logic exactly:
                # tmp_list = [smiles]+[Chem.MolToSmiles(x) for x in pair_list]+[name, pIC50]
                # Note: FragmentMol usually returns [Core, R-group] or similar.
                # The notebook logic assumes unpacking into specific columns later.
                
                # Check list length to avoid errors if fragmentation yields unexpected counts
                cores_and_r = [Chem.MolToSmiles(x) for x in pair_list]
                # We need exactly 2 parts for the dataframe columns ["Core", "R_group"]
                if len(cores_and_r) == 2:
                    tmp_list = [smiles, cores_and_r[0], cores_and_r[1], name, pIC50]
                    row_list.append(tmp_list)
        
        if i % 10 == 0:
            progress_bar.progress(min(i / total_rows, 1.0))
            
    progress_bar.empty()
    
    row_df = pd.DataFrame(row_list, columns=["SMILES", "Core", "R_group", "Name", "pIC50"])
    return row_df

@st.cache_data
def find_pairs(row_df):
    """Step 2: Collect Pairs with Same Scaffold"""
    delta_list = []
    
    grouped = row_df.groupby("Core")
    progress_bar = st.progress(0)
    total_groups = len(grouped)
    
    for i, (k, v) in enumerate(grouped):
        if len(v) > 1: # Notebook says > 2, but logic implies pairs (combinations of 2). >1 is sufficient for combinations.
            for a, b in combinations(range(0, len(v)), 2):
                reagent_a = v.iloc[a]
                reagent_b = v.iloc[b]
                
                if reagent_a.SMILES == reagent_b.SMILES:
                    continue
                
                # Canonical ordering
                reagent_a, reagent_b = sorted([reagent_a, reagent_b], key=lambda x: x.SMILES)
                
                delta = reagent_b.pIC50 - reagent_a.pIC50
                
                transform = f"{reagent_a.R_group.replace('*','*-')}>>{reagent_b.R_group.replace('*','*-')}"
                
                delta_list.append(
                    list(reagent_a.values) + 
                    list(reagent_b.values) + 
                    [transform, delta]
                )
        if i % 50 == 0:
             progress_bar.progress(min(i / total_groups, 1.0))

    progress_bar.empty()

    cols = ["SMILES_1", "Core_1", "R_group_1", "Name_1", "pIC50_1",
            "SMILES_2", "Core_2", "Rgroup_2", "Name_2", "pIC50_2", # Note: Notebook had typo "Rgroup_1" for second one? Correcting to context or keeping literal.
            "Transform", "Delta"]
    
    # Notebook column list check: 
    # cols = ["SMILES_1","Core_1","R_group_1","Name_1","pIC50_1", "SMILES_2","Core_2","Rgroup_1","Name_2","pIC50_2", "Transform","Delta"]
    # The notebook actually named the second R-group column "Rgroup_1" (typo in original?). I will name it Rgroup_2 for clarity but it maps to the second reagent's data.
    
    delta_df = pd.DataFrame(delta_list, columns=cols)
    return delta_df

@st.cache_data
def analyze_transforms(delta_df, min_occurrence=5):
    """Step 3: Summarize Transforms"""
    mmp_list = []
    
    for k, v in delta_df.groupby("Transform"):
        if len(v) > min_occurrence:
            mmp_list.append([k, len(v), v.Delta.values])
            
    mmp_df = pd.DataFrame(mmp_list, columns=["Transform", "Count", "Deltas"])
    
    if mmp_df.empty:
        return mmp_df
        
    mmp_df['idx'] = range(0, len(mmp_df))
    mmp_df['mean_delta'] = mmp_df['Deltas'].apply(lambda x: x.mean())
    
    # Reaction Smart Generation
    # Note: We cannot pickle RDKit objects easily in cache, so we generate them on the fly or store as binary/smarts
    # We will store the SMARTS string, and generate Mol objects only when displaying.
    
    return mmp_df

# ---------------------------------------------------------
# 4. STREAMLIT UI LAYOUT
# ---------------------------------------------------------

st.title("MMP Analysis App")
st.markdown("""
This app replicates the Matched Molecular Pair (MMP) analysis process.
Upload your dataset to identify structural transformations influencing pIC50 values.
""")

# Sidebar Inputs
with st.sidebar:
    st.header("Settings")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file:
        # Quick peek to let user select columns
        df_preview = pd.read_csv(uploaded_file, nrows=5)
        cols = df_preview.columns.tolist()
        
        smiles_col = st.selectbox("SMILES Column", cols, index=cols.index("SMILES") if "SMILES" in cols else 0)
        name_col = st.selectbox("Name/ID Column", cols, index=cols.index("Name") if "Name" in cols else 0)
        act_col = st.selectbox("Activity (pIC50) Column", cols, index=cols.index("pIC50") if "pIC50" in cols else 0)
        
        min_occurrence = st.number_input("Min Transform Occurrence", value=5, min_value=1)
        run_btn = st.button("Run Analysis")

# Main Execution
if uploaded_file and run_btn:
    # Load Data
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file)
    
    st.info(f"Loaded {len(df)} molecules. Starting Fragmentation...")
    
    # 1. Process Fragments
    row_df = process_data(df, smiles_col, name_col, act_col)
    st.write(f"Generated {len(row_df)} fragments.")
    
    # 2. Find Pairs
    st.info("Finding Matched Pairs...")
    delta_df = find_pairs(row_df)
    st.write(f"Found {len(delta_df)} matched pairs.")
    
    # 3. Analyze
    st.info("Analyzing Transformations...")
    mmp_df = analyze_transforms(delta_df, min_occurrence)
    
    if mmp_df.empty:
        st.warning("No transformations found matching criteria.")
    else:
        # Create dictionary linking Transform string to Index (for later lookups)
        # Note: In Streamlit, we just filter delta_df directly by Transform string.
        
        st.success(f"Identified {len(mmp_df)} unique transformations.")

        # ---------------------------------------------------------
        # RESULTS DISPLAY
        # ---------------------------------------------------------
        
        tab1, tab2, tab3 = st.tabs(["Top Transformations", "All Transforms Table", "Downloads"])

        with tab1:
            st.subheader("Top Positive Transformations (Potency Enhancers)")
            top_pos = mmp_df.sort_values("mean_delta", ascending=False).head(3)
            
            for i, (idx, row) in enumerate(top_pos.iterrows()):
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown(f"**Rank {i+1}:** `{row['Transform']}`")
                    st.write(f"**Mean ΔpIC50:** {row['mean_delta']:.2f}")
                    st.write(f"**Count:** {row['Count']}")
                    
                    # Generate Reaction Image
                    rxn = AllChem.ReactionFromSmarts(row['Transform'].replace('*-','*'), useSmiles=True)
                    st.image(rxn_to_image(rxn), use_container_width=True)
                    
                with col2:
                    # Strip Plot
                    fig, ax = plt.subplots(figsize=(4, 2))
                    sns.stripplot(x=row['Deltas'], ax=ax, color='blue', alpha=0.6, jitter=0.2)
                    ax.axvline(0, ls="--", c="red")
                    ax.set_xlim(-5, 5)
                    ax.set_title("ΔpIC50 Distribution")
                    st.pyplot(fig)
                    plt.close(fig)

                # Show examples for this transform
                with st.expander(f"Show Compound Pairs for Rank {i+1}"):
                    examples = delta_df[delta_df['Transform'] == row['Transform']].head(6)
                    
                    # Prepare for mols2grid
                    # We want to show pair: SMILES_1 -> SMILES_2. 
                    # Mols2grid shows single mols. We can construct a "reaction" smiles or just show them side by side.
                    # The notebook shows a grid of separate molecules.
                    # We will create a list of mols to display: Mol A and Mol B alternating.
                    
                    grid_data = []
                    for _, ex_row in examples.iterrows():
                        grid_data.append({
                            "SMILES": ex_row['SMILES_1'], 
                            "Name": f"{ex_row['Name_1']} (A)", 
                            "pIC50": ex_row['pIC50_1'],
                            "Type": "Reagent A"
                        })
                        grid_data.append({
                            "SMILES": ex_row['SMILES_2'], 
                            "Name": f"{ex_row['Name_2']} (B)", 
                            "pIC50": ex_row['pIC50_2'],
                            "Type": "Reagent B"
                        })
                    
                    grid_df = pd.DataFrame(grid_data)
                    raw_html = mols2grid.display(
                        grid_df, 
                        smiles_col="SMILES", 
                        subset=["img", "Name", "pIC50"],
                        n_cols=4, 
                        size=(150, 150)
                    )._repr_html_()
                    components.html(raw_html, height=400, scrolling=True)

            st.markdown("---")
            st.subheader("Top Negative Transformations (Potency Diminishers)")
            top_neg = mmp_df.sort_values("mean_delta", ascending=True).head(3)
            
            for i, (idx, row) in enumerate(top_neg.iterrows()):
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown(f"**Rank {i+1}:** `{row['Transform']}`")
                    st.write(f"**Mean ΔpIC50:** {row['mean_delta']:.2f}")
                    st.write(f"**Count:** {row['Count']}")
                    rxn = AllChem.ReactionFromSmarts(row['Transform'].replace('*-','*'), useSmiles=True)
                    st.image(rxn_to_image(rxn), use_container_width=True)
                with col2:
                    fig, ax = plt.subplots(figsize=(4, 2))
                    sns.stripplot(x=row['Deltas'], ax=ax, color='red', alpha=0.6, jitter=0.2)
                    ax.axvline(0, ls="--", c="red")
                    ax.set_xlim(-5, 5)
                    st.pyplot(fig)
                    plt.close(fig)

        with tab2:
            st.subheader("Full Transformation Table")
            st.dataframe(mmp_df.drop(columns=['Deltas', 'idx'], errors='ignore').sort_values("mean_delta", ascending=False))
            
            st.write("### Interactive Detail Viewer")
            selected_transform = st.selectbox("Select a Transform to view details:", mmp_df['Transform'].unique())
            
            if selected_transform:
                sel_row = mmp_df[mmp_df['Transform'] == selected_transform].iloc[0]
                st.write(f"**Mean Delta:** {sel_row['mean_delta']:.2f}, **Count:** {sel_row['Count']}")
                
                rxn = AllChem.ReactionFromSmarts(selected_transform.replace('*-','*'), useSmiles=True)
                st.image(rxn_to_image(rxn), width=400)
                
                # Distribution
                fig, ax = plt.subplots(figsize=(6, 2))
                sns.stripplot(x=sel_row['Deltas'], ax=ax, color='purple', alpha=0.6)
                ax.axvline(0, ls="--", c="black")
                ax.set_title("ΔpIC50 Distribution")
                st.pyplot(fig)
                plt.close(fig)

        with tab3:
            st.subheader("Export Results")
            
            # Split into 4 parts like the notebook
            mmp_sorted = mmp_df.sort_values("mean_delta")
            # Convert 'Deltas' (array) to string for Excel export to avoid errors
            mmp_export = mmp_sorted.copy()
            mmp_export['Deltas'] = mmp_export['Deltas'].apply(lambda x: str(list(x)))
            
            # Generate Excel file in memory
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                mmp_export.to_excel(writer, index=False, sheet_name='All Transforms')
                delta_df.to_excel(writer, index=False, sheet_name='Pairs Data')
            
            st.download_button(
                label="Download Full Analysis (Excel)",
                data=output.getvalue(),
                file_name="MMP_Analysis_Full.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

else:
    st.info("Please upload a CSV file to begin.")

