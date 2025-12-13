import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import Draw
import io
import base64
from operator import itemgetter
from itertools import combinations
import mols2grid
import seaborn as sns
import matplotlib.pyplot as plt
import subprocess
import sys

# --- Page configuration ---
st.set_page_config(
    page_title="MMP Analysis Tool",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Install necessary packages ---
def install_package(package):
    """Install a package if not already installed"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])
        return True
    except Exception:
        return False

# List of required packages
required_packages = [
    ("streamlit", "streamlit"),
    ("pandas", "pandas"),
    ("rdkit", "rdkit-pypi"),
    ("mols2grid", "mols2grid"),
    ("seaborn", "seaborn"),
    ("matplotlib", "matplotlib"),
]

# Check and install packages
with st.spinner("Checking and installing required packages..."):
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            st.info(f"Installing {package_name}...")
            success = install_package(package_name)
            if not success:
                st.error(f"Failed to install {package_name}. Please install manually.")
                st.stop()

# --- Custom functions ---
def get_largest_fragment(mol):
    """Extract the largest fragment from a molecule"""
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True)
    if not frags:
        return None
    num_atoms = [x.GetNumHeavyAtoms() for x in frags]
    idx = num_atoms.index(max(num_atoms))
    return frags[idx]

def remove_map_nums(mol):
    """Remove atom mapping numbers from a molecule"""
    if mol is None:
        return
    for atm in mol.GetAtoms():
        atm.SetAtomMapNum(0)

def sort_fragments(mol):
    """Sort fragments by number of atoms (descending)"""
    if mol is None:
        return []
    frag_list = list(Chem.GetMolFrags(mol, asMols=True))
    [remove_map_nums(x) for x in frag_list]
    frag_num_atoms_list = [(x.GetNumAtoms(), x) for x in frag_list]
    frag_num_atoms_list.sort(key=itemgetter(0), reverse=True)
    return [x[1] for x in frag_num_atoms_list]

def FragmentMol(mol, maxCuts=1, minMolSize=1, linker_smarts="[D3-D4;!R]"):
    """Fragment molecule at linker positions"""
    # Remove stereochemistry for fragmentation
    clean_mol = Chem.Mol(mol)
    Chem.RemoveStereochemistry(clean_mol)
    clean_mol = Chem.AddHs(clean_mol)
    
    # If maxCuts is 0, just return the original molecule
    if maxCuts == 0:
        return [(0, clean_mol)]

    res = []
    # Find all possible linker cut sites
    patt = Chem.MolFromSmarts(linker_smarts)
    if patt is None:
        return [(0, clean_mol)]
    
    matches = clean_mol.GetSubstructMatches(patt)
    cut_bonds = set()
    for m in matches:
        for atm_idx in m:
            atm = clean_mol.GetAtomWithIdx(atm_idx)
            for bond in atm.GetBonds():
                if not bond.IsInRing():  # Exclude cutting bonds in a ring
                    cut_bonds.add(tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))))

    if not cut_bonds:
        return [(0, clean_mol)]

    # Try all combinations of cuts up to maxCuts
    for num_cuts in range(1, maxCuts + 1):
        if num_cuts > len(cut_bonds):
            continue
        for combo_bonds in combinations(cut_bonds, num_cuts):
            bonds_to_cut = []
            for b_atm1, b_atm2 in combo_bonds:
                bond = clean_mol.GetBondBetweenAtoms(b_atm1, b_atm2)
                if bond:
                    bonds_to_cut.append(bond.GetIdx())
            
            if bonds_to_cut:
                tmp = Chem.FragmentOnBonds(clean_mol, bonds_to_cut, addDummies=True)
                res.append((num_cuts, tmp))
    return res

# --- Image generation helper functions ---
def rxn_to_base64_image(rxn):
    """Convert RDKit reaction to base64 encoded image"""
    if rxn is None:
        return ""
    try:
        img = Draw.ReactionToImage(rxn, subImgSize=(300, 150))
        img_buffer = io.BytesIO()
        img.save(img_buffer, format="PNG")
        im_text64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
        img_str = f'<img src="data:image/png;base64,{im_text64}" style="width: 300px; height: 150px;"/>'
        return img_str
    except Exception as e:
        return f"Error rendering reaction: {str(e)}"

def stripplot_base64_image(dist):
    """Create stripplot of delta values as base64 encoded image"""
    if not dist:
        return ""
    
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(figsize=(4, 1.5))
    sns.stripplot(x=dist, ax=ax, alpha=0.6, size=8, jitter=0.3)
    ax.axvline(0, ls="--", c="red", linewidth=1)
    ax.set_xlim(-5, 5)
    ax.set_yticks([])
    ax.set_xlabel("ΔpIC50", fontsize=10)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    
    s = io.BytesIO()
    plt.savefig(s, format='png', bbox_inches="tight", dpi=100)
    plt.close(fig)
    s = base64.b64encode(s.getvalue()).decode("utf-8")
    return f'<img src="data:image/png;base64,{s}" style="width: 400px; height: 150px;">'

def find_examples(delta_df, query_idx, transform_smarts):
    """Find example pairs for a specific transformation"""
    example_list = []
    if query_idx is None or delta_df.empty:
        return pd.DataFrame(columns=["SMILES", "Name", "pIC50"])
    
    # Filter by transformation SMARTS instead of idx
    filtered_df = delta_df[delta_df["Transform"] == transform_smarts]
    
    if filtered_df.empty:
        return pd.DataFrame(columns=["SMILES", "Name", "pIC50"])
    
    # Get unique examples
    seen_smiles = set()
    for _, row in filtered_df.iterrows():
        for prefix in ["1", "2"]:
            smi = row[f"SMILES_{prefix}"]
            if smi not in seen_smiles:
                seen_smiles.add(smi)
                example_list.append({
                    "SMILES": smi,
                    "Name": row[f"Name_{prefix}"],
                    "pIC50": row[f"pIC50_{prefix}"]
                })
    
    return pd.DataFrame(example_list)

# --- Main Streamlit App ---
def main():
    st.title("🧪 Matched Molecular Pairs (MMPs) Analysis")
    
    st.markdown("""
    This application performs Matched Molecular Pairs (MMPs) analysis on chemical compound datasets.
    It identifies pairs of molecules that differ by a small chemical transformation and analyzes the
    impact on biological activity (pIC50).
    """)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        min_transform_occurrence = st.slider(
            "Minimum transform occurrence:",
            min_value=1,
            max_value=20,
            value=5,
            help="Minimum number of times a transformation must occur to be included"
        )
        max_cuts = st.slider(
            "Maximum cuts per molecule:",
            min_value=1,
            max_value=3,
            value=1,
            help="Maximum number of cuts for fragmentation"
        )
        
        st.header("📊 Display Options")
        show_raw_data = st.checkbox("Show raw data", value=False)
        num_preview_rows = st.slider("Preview rows:", 5, 50, 10)
    
    # 1. Load Data
    st.header("1. 📁 Load Data")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "Upload your CSV file",
            type="csv",
            help="CSV must contain 'SMILES', 'Name', and 'pIC50' columns"
        )
    
    with col2:
        st.markdown("### Sample Data Format")
        sample_df = pd.DataFrame({
            "SMILES": ["CC(=O)Oc1ccccc1C(=O)O", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"],
            "Name": ["Aspirin", "Ibuprofen"],
            "pIC50": [5.0, 5.5]
        })
        st.dataframe(sample_df, hide_index=True, use_container_width=True)
    
    # Load data
    df = None
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Data loaded successfully! ({len(df)} rows, {len(df.columns)} columns)")
            
            if show_raw_data:
                with st.expander("📋 Raw Data Preview"):
                    st.dataframe(df.head(num_preview_rows), use_container_width=True)
                    
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.stop()
    else:
        st.info("👆 Please upload a CSV file to begin analysis")
        st.stop()
    
    # Validate required columns
    required_cols = ['SMILES', 'Name', 'pIC50']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
        st.stop()
    
    # 2. Preprocessing
    st.header("2. 🔧 Preprocessing Molecules")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Create RDKit Mol objects
    status_text.text("Converting SMILES to RDKit molecules...")
    mols = []
    for i, smiles in enumerate(df.SMILES):
        mol = Chem.MolFromSmiles(str(smiles))
        mols.append(mol)
        progress_bar.progress((i + 1) / len(df) * 0.33)
    
    df['mol'] = mols
    
    # Filter out invalid molecules
    df_valid = df.dropna(subset=['mol']).copy()
    if len(df_valid) < len(df):
        st.warning(f"⚠️ Removed {len(df) - len(df_valid)} invalid molecules")
    
    if df_valid.empty:
        st.error("❌ No valid molecules found. Please check your SMILES strings.")
        st.stop()
    
    # Get largest fragment
    status_text.text("Extracting largest fragments...")
    largest_frags = []
    for i, mol in enumerate(df_valid['mol']):
        largest_frags.append(get_largest_fragment(mol))
        progress_bar.progress(0.33 + (i + 1) / len(df_valid) * 0.33)
    
    df_valid['mol'] = largest_frags
    
    # Remove molecules that became None after fragmentation
    df_valid = df_valid.dropna(subset=['mol']).copy()
    
    status_text.text("Preprocessing complete!")
    progress_bar.progress(1.0)
    
    st.success(f"✅ {len(df_valid)} valid molecules ready for analysis")
    
    # 3. Decompose Molecules
    st.header("3. ✂️ Decompose Molecules")
    
    decompose_status = st.empty()
    decompose_progress = st.progress(0)
    
    row_list = []
    for i, row in enumerate(df_valid.itertuples()):
        decompose_status.text(f"Processing molecule {i+1}/{len(df_valid)}")
        mol = row.mol
        if mol is None:
            continue
        
        frag_list = FragmentMol(mol, maxCuts=max_cuts)
        for _, frag_mol in frag_list:
            if frag_mol is None:
                continue
            pair_list = sort_fragments(frag_mol)
            
            if len(pair_list) >= 2:
                core_smiles = Chem.MolToSmiles(pair_list[0]) if pair_list[0] else None
                r_group_smiles = Chem.MolToSmiles(pair_list[1]) if len(pair_list) > 1 and pair_list[1] else None
                
                if core_smiles:
                    row_list.append([
                        row.SMILES,
                        core_smiles,
                        r_group_smiles,
                        row.Name,
                        row.pIC50
                    ])
        
        decompose_progress.progress((i + 1) / len(df_valid))
    
    decompose_status.empty()
    decompose_progress.empty()
    
    if not row_list:
        st.error("❌ No fragments generated. Try adjusting fragmentation parameters.")
        st.stop()
    
    row_df = pd.DataFrame(row_list, columns=["SMILES", "Core", "R_group", "Name", "pIC50"])
    
    with st.expander("📋 Decomposition Results"):
        st.dataframe(row_df.head(num_preview_rows), use_container_width=True)
        st.caption(f"Total fragments: {len(row_df)}")
    
    # 4. Collect Matched Molecular Pairs
    st.header("4. 🔗 Collect Matched Molecular Pairs")
    
    collect_status = st.empty()
    collect_progress = st.progress(0)
    
    delta_list = []
    grouped_cores = list(row_df.groupby("Core"))
    
    for i, (core, group) in enumerate(grouped_cores):
        collect_status.text(f"Processing core {i+1}/{len(grouped_cores)} ({len(group)} molecules)")
        
        if len(group) >= 2:
            # Create all unique pairs
            for a, b in combinations(group.index, 2):
                reagent_a = group.loc[a]
                reagent_b = group.loc[b]
                
                # Skip if same molecule
                if reagent_a.SMILES == reagent_b.SMILES:
                    continue
                
                # Sort for consistent ordering
                if str(reagent_a.SMILES) > str(reagent_b.SMILES):
                    reagent_a, reagent_b = reagent_b, reagent_a
                
                # Skip if R_group is missing
                if pd.isna(reagent_a.R_group) or pd.isna(reagent_b.R_group):
                    continue
                
                # Calculate delta
                delta = float(reagent_b.pIC50) - float(reagent_a.pIC50)
                
                # Create transform SMARTS
                transform_smarts = f"{reagent_a.R_group.replace('*', '*-')}>>{reagent_b.R_group.replace('*', '*-')}"
                
                delta_list.append([
                    reagent_a.SMILES, reagent_a.Core, reagent_a.R_group, reagent_a.Name, reagent_a.pIC50,
                    reagent_b.SMILES, reagent_b.Core, reagent_b.R_group, reagent_b.Name, reagent_b.pIC50,
                    transform_smarts, delta
                ])
        
        collect_progress.progress((i + 1) / len(grouped_cores))
    
    collect_status.empty()
    collect_progress.empty()
    
    if not delta_list:
        st.error("❌ No matched molecular pairs found. Try adjusting parameters.")
        st.stop()
    
    cols = ["SMILES_1", "Core_1", "R_group_1", "Name_1", "pIC50_1",
            "SMILES_2", "Core_2", "R_group_2", "Name_2", "pIC50_2",
            "Transform", "Delta"]
    
    delta_df = pd.DataFrame(delta_list, columns=cols)
    
    st.success(f"✅ Found {len(delta_df)} matched molecular pairs")
    
    with st.expander("📋 MMP Pairs Preview"):
        st.dataframe(delta_df.head(num_preview_rows), use_container_width=True)
    
    # 5. Summarize Transformations
    st.header("5. 📊 Summarize Transformations")
    
    # Group by transformation
    transform_groups = delta_df.groupby("Transform").agg({
        "Delta": list,
        "Transform": "count"
    }).rename(columns={"Transform": "Count", "Delta": "Deltas"})
    
    # Filter by minimum occurrence
    transform_groups = transform_groups[transform_groups["Count"] >= min_transform_occurrence]
    
    if transform_groups.empty:
        st.warning(f"⚠️ No transformations found with minimum occurrence of {min_transform_occurrence}")
        st.stop()
    
    # Create summary dataframe
    mmp_df = pd.DataFrame({
        "Transform": transform_groups.index,
        "Count": transform_groups["Count"].values,
        "Deltas": transform_groups["Deltas"].values
    })
    
    # Calculate statistics
    mmp_df["mean_delta"] = mmp_df["Deltas"].apply(lambda x: sum(x) / len(x))
    mmp_df["std_delta"] = mmp_df["Deltas"].apply(lambda x: pd.Series(x).std() if len(x) > 1 else 0)
    mmp_df["min_delta"] = mmp_df["Deltas"].apply(min)
    mmp_df["max_delta"] = mmp_df["Deltas"].apply(max)
    mmp_df["idx"] = range(len(mmp_df))
    
    # Create reaction objects and images
    with st.spinner("Generating reaction diagrams..."):
        mmp_df["rxn_mol"] = mmp_df["Transform"].apply(
            lambda x: AllChem.ReactionFromSmarts(x.replace('*-', '*'), useSmiles=True)
        )
        mmp_df["MMP Transform"] = mmp_df["rxn_mol"].apply(rxn_to_base64_image)
        mmp_df["Delta Distribution"] = mmp_df["Deltas"].apply(stripplot_base64_image)
    
    # Sort options
    col1, col2 = st.columns(2)
    with col1:
        sort_by = st.selectbox(
            "Sort by:",
            options=["mean_delta", "Count", "std_delta"],
            format_func=lambda x: {
                "mean_delta": "Mean ΔpIC50",
                "Count": "Frequency",
                "std_delta": "Standard Deviation"
            }[x]
        )
    
    with col2:
        sort_ascending = st.checkbox("Sort ascending", value=False)
    
    # Sort and display
    mmp_df_sorted = mmp_df.sort_values(sort_by, ascending=sort_ascending)
    
    st.subheader(f"Top {len(mmp_df_sorted)} Transformations")
    
    # Create HTML table for display
    display_df = mmp_df_sorted.copy()
    display_df["Mean ΔpIC50"] = display_df["mean_delta"].round(3)
    display_df["Frequency"] = display_df["Count"]
    display_df["Std Dev"] = display_df["std_delta"].round(3)
    
    cols_to_display = ["MMP Transform", "Frequency", "Mean ΔpIC50", "Std Dev", "Delta Distribution"]
    
    html_table = display_df[cols_to_display].to_html(escape=False, index=False)
    
    # Add some CSS styling
    styled_html = f"""
    <style>
    .dataframe {{
        width: 100%;
    }}
    .dataframe td {{
        vertical-align: middle;
        padding: 10px;
    }}
    .dataframe tr:nth-child(even) {{
        background-color: #f5f5f5;
    }}
    </style>
    {html_table}
    """
    
    st.markdown(styled_html, unsafe_allow_html=True)
    
    # 6. Explore Specific Transformations
    st.header("6. 🔍 Explore Specific Transformations")
    
    if not mmp_df_sorted.empty:
        # Select transformation
        transform_options = mmp_df_sorted.apply(
            lambda row: f"{row['Transform'][:50]}... (Δ={row['mean_delta']:.2f}, n={row['Count']})",
            axis=1
        ).tolist()
        
        selected_idx = st.selectbox(
            "Select a transformation to explore:",
            options=mmp_df_sorted["idx"].tolist(),
            format_func=lambda x: transform_options[mmp_df_sorted[mmp_df_sorted["idx"] == x].index[0]]
        )
        
        if selected_idx is not None:
            selected_row = mmp_df_sorted[mmp_df_sorted["idx"] == selected_idx].iloc[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Transformation Details")
                st.markdown(f"**SMARTS:** `{selected_row['Transform']}`")
                st.markdown(f"**Frequency:** {selected_row['Count']} occurrences")
                st.markdown(f"**Mean ΔpIC50:** {selected_row['mean_delta']:.3f}")
                st.markdown(f"**Std Dev:** {selected_row['std_delta']:.3f}")
                st.markdown(f"**Range:** [{selected_row['min_delta']:.3f}, {selected_row['max_delta']:.3f}]")
            
            with col2:
                st.subheader("Reaction Diagram")
                st.markdown(selected_row['MMP Transform'], unsafe_allow_html=True)
            
            st.subheader("ΔpIC50 Distribution")
            st.markdown(selected_row['Delta Distribution'], unsafe_allow_html=True)
            
            # Show example compounds
            st.subheader("Example Compounds")
            example_df = find_examples(delta_df, selected_idx, selected_row["Transform"])
            
            if not example_df.empty:
                # Prepare dataframe for mols2grid
                example_df_display = example_df.copy()
                example_df_display = example_df_display.rename(columns={
                    "SMILES": "smiles",
                    "Name": "Name",
                    "pIC50": "pIC50"
                })
                
                # Display with mols2grid
                try:
                    grid = mols2grid.display(
                        example_df_display,
                        smiles_col="smiles",
                        subset=["img", "Name", "pIC50"],
                        transform={"pIC50": lambda x: f"{x:.2f}"},
                        n_cols=4,
                        size=(200, 150),
                        selection=False
                    )
                    
                    html_content = grid._repr_html_()
                    st.components.v1.html(html_content, height=600, scrolling=False)
                    
                except Exception as e:
                    st.error(f"Error displaying molecules: {str(e)}")
                    st.dataframe(example_df, use_container_width=True)
            else:
                st.info("No example compounds found for this transformation.")
    
    # Summary statistics
    st.header("📈 Summary Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Molecules", len(df_valid))
        st.metric("Valid SMILES", f"{len(df_valid)/len(df)*100:.1f}%")
    
    with col2:
        st.metric("Total Pairs", len(delta_df))
        st.metric("Unique Cores", len(grouped_cores))
    
    with col3:
        st.metric("Frequent Transforms", len(mmp_df))
        if not mmp_df.empty:
            avg_delta = mmp_df["mean_delta"].mean()
            st.metric("Avg ΔpIC50", f"{avg_delta:.3f}")
    
    # Download results
    st.header("💾 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Convert delta_df to CSV
        csv_delta = delta_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download MMP Pairs (CSV)",
            data=csv_delta,
            file_name="mmp_pairs_results.csv",
            mime="text/csv"
        )
    
    with col2:
        # Convert mmp_df to CSV (without image columns)
        mmp_df_csv = mmp_df.drop(columns=["rxn_mol", "MMP Transform", "Delta Distribution"]).to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Transformations (CSV)",
            data=mmp_df_csv,
            file_name="mmp_transformations.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
