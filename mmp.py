import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMMPA import FragmentMol
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor
from itertools import combinations
from operator import itemgetter
from tqdm import tqdm

# Set page config
st.set_page_config(page_title="Systematic MMP Analysis Tool", layout="wide", page_icon="🧪")

# ------------------------------------------------------------------------------
# 1. HELPER FUNCTIONS (Based on your provided snippets)
# ------------------------------------------------------------------------------

def remove_map_nums(mol):
    """Remove atom map numbers from a molecule."""
    for atm in mol.GetAtoms():
        atm.SetAtomMapNum(0)
    return mol

def sort_fragments(mol):
    """
    Transform a molecule with multiple fragments into a list of molecules 
    that is sorted by number of atoms from largest to smallest.
    """
    frag_list = list(Chem.GetMolFrags(mol, asMols=True))
    [remove_map_nums(x) for x in frag_list]
    frag_num_atoms_list = [(x.GetNumAtoms(), x) for x in frag_list]
    frag_num_atoms_list.sort(key=itemgetter(0), reverse=True)
    return [x[1] for x in frag_num_atoms_list]

def rxn_to_base64_image(rxn_smarts):
    """Convert a transform SMARTS string to a base64 encoded image."""
    try:
        # Convert SMARTS transform back to reaction object
        rxn = AllChem.ReactionFromSmarts(rxn_smarts.replace('*-', '*'), useSmiles=True)
        if rxn is None:
            return ""
        
        drawer = rdMolDraw2D.MolDraw2DCairo(300, 150)
        drawer.DrawReaction(rxn)
        drawer.FinishDrawing()
        
        img_bytes = drawer.GetDrawingText()
        im_text64 = base64.b64encode(img_bytes).decode('utf8')
        return f'<img src="data:image/png;base64, {im_text64}"/>'
    except Exception as e:
        return f"Error: {str(e)}"

def stripplot_base64_image(dist):
    """Plot a distribution as a seaborn stripplot and return base64 image."""
    try:
        plt.figure(figsize=(4, 1.5))
        sns.set_style('whitegrid')
        
        # Create strip plot
        ax = sns.stripplot(x=dist, jitter=0.2, alpha=0.6, color="#1f77b4")
        
        # Add mean line
        mean_val = np.mean(dist)
        ax.axvline(mean_val, ls="--", c="red", alpha=0.8)
        ax.axvline(0, ls="-", c="black", alpha=0.3)
        
        # Dynamic limits based on data range, ensuring 0 is included
        min_val, max_val = min(dist), max(dist)
        margin = (max_val - min_val) * 0.1 if max_val != min_val else 1.0
        ax.set_xlim(min(min(dist), -1) - margin, max(max(dist), 1) + margin)
        
        # Clean up axes
        ax.set_yticks([])
        ax.set_xlabel("ΔpIC50", fontsize=8)
        sns.despine(left=True)
        
        # Save to buffer
        s = io.BytesIO()
        plt.savefig(s, format='png', bbox_inches="tight", dpi=100)
        plt.close()
        
        s = base64.b64encode(s.getvalue()).decode("utf-8").replace("\n", "")
        return f'<img src="data:image/png;base64,{s}">'
    except Exception:
        return ""

# ------------------------------------------------------------------------------
# 2. CORE LOGIC
# ------------------------------------------------------------------------------

def process_file(uploaded_file):
    """Process uploaded CSV and return basic dataframe."""
    df = pd.read_csv(uploaded_file)
    
    # Standardize column names
    cols = [c.lower() for c in df.columns]
    if 'smiles' not in cols or 'pic50' not in cols:
        st.error("CSV must contain 'SMILES' and 'pIC50' columns.")
        return None
        
    # Rename columns to standard format
    df.rename(columns={
        df.columns[cols.index('smiles')]: 'SMILES', 
        df.columns[cols.index('pic50')]: 'pIC50'
    }, inplace=True)
    
    if 'Name' not in df.columns:
        df['Name'] = [f"Cmp_{i}" for i in range(len(df))]
        
    # Create Mol objects
    PandasTools = None  # Avoiding dependency if not strictly needed, using apply
    df['mol'] = df['SMILES'].apply(Chem.MolFromSmiles)
    
    # Drop invalid mols
    df = df.dropna(subset=['mol'])
    
    return df

def perform_mmp_analysis(df, min_transform_occurrence):
    """Run the MMP analysis workflow."""
    
    # 1. Fragment Generation
    row_list = []
    
    # Progress bar
    progress_text = "Generating fragments..."
    my_bar = st.progress(0, text=progress_text)
    
    total_mols = len(df)
    
    for idx, (smiles, name, pIC50, mol) in enumerate(df[['SMILES', 'Name', 'pIC50', 'mol']].values):
        # Update progress
        if idx % 10 == 0:
            my_bar.progress(int(idx / total_mols * 30), text=f"Fragmenting molecule {idx+1}/{total_mols}")

        # Exhaustive single cut (maxCuts=1)
        frag_list = FragmentMol(mol, maxCuts=1)
        
        for _, frag_mol in frag_list:
            pair_list = sort_fragments(frag_mol)
            
            # We only care about single cuts resulting in exactly 2 pieces (Core + R-group)
            if len(pair_list) == 2:
                core_mol, rgroup_mol = pair_list[0], pair_list[1]
                
                # Convert to SMILES
                core_smi = Chem.MolToSmiles(core_mol)
                rgroup_smi = Chem.MolToSmiles(rgroup_mol)
                
                tmp_list = [smiles, core_smi, rgroup_smi, name, pIC50]
                row_list.append(tmp_list)

    row_df = pd.DataFrame(row_list, columns=["SMILES", "Core", "R_group", "Name", "pIC50"])
    
    if row_df.empty:
        return None, None

    # 2. Pair Generation
    delta_list = []
    unique_cores = row_df['Core'].unique()
    total_cores = len(unique_cores)
    
    my_bar.progress(30, text="Generating molecular pairs...")
    
    for idx, (core, group) in enumerate(row_df.groupby("Core")):
        if idx % 50 == 0:
             my_bar.progress(30 + int(idx / total_cores * 40), text=f"Processing core {idx+1}/{total_cores}")
             
        if len(group) > 1: # Need at least 2 compounds to make a pair
            # Use combinations to get unique pairs
            for a, b in combinations(range(len(group)), 2):
                reagent_a = group.iloc[a]
                reagent_b = group.iloc[b]
                
                # Skip self-pairs (though combinations handles indices, check smiles to be safe)
                if reagent_a.SMILES == reagent_b.SMILES:
                    continue
                
                # Sort by SMILES to ensure consistent direction (or sort by pIC50 if preferred)
                # Here we strictly follow the snippet: sort by SMILES
                compounds = sorted([reagent_a, reagent_b], key=lambda x: x.SMILES)
                ra, rb = compounds[0], compounds[1]
                
                delta = rb.pIC50 - ra.pIC50
                
                # Create Transform String
                transform = f"{ra.R_group.replace('*', '*-')}>>{rb.R_group.replace('*', '*-')}"
                
                # Store data
                delta_list.append({
                    'Core': core,
                    'SMILES_A': ra.SMILES, 'Name_A': ra.Name, 'pIC50_A': ra.pIC50, 'R_group_A': ra.R_group,
                    'SMILES_B': rb.SMILES, 'Name_B': rb.Name, 'pIC50_B': rb.pIC50, 'R_group_B': rb.R_group,
                    'Transform': transform,
                    'Delta': delta
                })
    
    delta_df = pd.DataFrame(delta_list)
    
    if delta_df.empty:
        return row_df, None

    # 3. Aggregation and Stats
    my_bar.progress(80, text="Calculating statistics...")
    
    mmp_list = []
    for transform, group in delta_df.groupby("Transform"):
        if len(group) >= min_transform_occurrence:
            deltas = group['Delta'].values
            mmp_list.append({
                'Transform': transform,
                'Count': len(group),
                'Mean_Delta': np.mean(deltas),
                'Std_Delta': np.std(deltas),
                'Deltas': deltas  # Keep array for plotting
            })
            
    mmp_df = pd.DataFrame(mmp_list)
    
    if not mmp_df.empty:
        mmp_df = mmp_df.sort_values('Count', ascending=False)
        # Add index for reference
        mmp_df = mmp_df.reset_index(drop=True)
    
    my_bar.progress(100, text="Analysis Complete!")
    return delta_df, mmp_df

# ------------------------------------------------------------------------------
# 3. MAIN UI
# ------------------------------------------------------------------------------

def main():
    st.title("🧪 Systematic MMP Analysis Tool")
    
    st.markdown("""
    This tool performs Matched Molecular Pair (MMP) analysis by:
    1.  **Fragmenting** all input molecules (Exhaustive Single Cut).
    2.  **Grouping** by common cores (No pre-filtering).
    3.  **Generating** all valid pairs and calculating ΔpIC50.
    4.  **Identifying** significant structural transformations.
    """)
    
    # Sidebar
    st.sidebar.header("Settings")
    
    # File Upload
    uploaded_file = st.sidebar.file_uploader("Upload CSV (SMILES, Name, pIC50)", type=["csv"])
    
    # Parameter: Min Occurrence
    min_occurrence = st.sidebar.number_input(
        "Min Transform Occurrence", 
        min_value=2, 
        value=3,
        help="Minimum number of pairs required to display a transform."
    )
    
    if uploaded_file is not None:
        try:
            df = process_file(uploaded_file)
            
            if df is not None:
                st.info(f"Loaded {len(df)} molecules successfully.")
                
                # Run Analysis Button
                if st.button("Run Analysis", type="primary"):
                    pairs_df, mmp_stats_df = perform_mmp_analysis(df, min_occurrence)
                    
                    if mmp_stats_df is not None and not mmp_stats_df.empty:
                        # --- RESULTS DISPLAY ---
                        
                        st.subheader("Significant Transforms")
                        
                        # Prepare data for display (add images)
                        # We use a copy for display to avoid pickling issues with images later if needed
                        display_df = mmp_stats_df.copy()
                        
                        # Generate Images (Progressive)
                        with st.spinner("Rendering images..."):
                            display_df['Transform Image'] = display_df['Transform'].apply(rxn_to_base64_image)
                            display_df['Distribution'] = display_df['Deltas'].apply(stripplot_base64_image)
                        
                        # HTML Table Construction
                        # We construct a custom HTML table because st.dataframe can't render base64 images easily
                        
                        html = '<table style="width:100%; border-collapse: collapse;">'
                        html += '<thead><tr style="background-color: #f0f2f6; text-align: left;">'
                        html += '<th style="padding: 10px;">Transform</th>'
                        html += '<th style="padding: 10px;">Visualization</th>'
                        html += '<th style="padding: 10px;">Count</th>'
                        html += '<th style="padding: 10px;">Mean ΔpIC50</th>'
                        html += '<th style="padding: 10px;">Distribution</th>'
                        html += '</tr></thead><tbody>'
                        
                        for _, row in display_df.iterrows():
                            html += f'<tr>'
                            html += f'<td style="padding: 10px; font-family: monospace; font-size: 0.9em;">{row["Transform"]}</td>'
                            html += f'<td style="padding: 10px;">{row["Transform Image"]}</td>'
                            html += f'<td style="padding: 10px;"><strong>{row["Count"]}</strong></td>'
                            html += f'<td style="padding: 10px;">{row["Mean_Delta"]:.2f} ± {row["Std_Delta"]:.2f}</td>'
                            html += f'<td style="padding: 10px;">{row["Distribution"]}</td>'
                            html += '</tr>'
                            
                        html += '</tbody></table>'
                        
                        st.markdown(html, unsafe_allow_html=True)
                        
                        # Downloads
                        st.divider()
                        st.subheader("Downloads")
                        
                        c1, c2 = st.columns(2)
                        
                        # 1. Pairs CSV
                        csv_pairs = pairs_df.drop(columns=['Deltas'], errors='ignore').to_csv(index=False)
                        c1.download_button(
                            "📥 Download All Molecular Pairs (.csv)",
                            csv_pairs,
                            "mmp_pairs.csv",
                            "text/csv"
                        )
                        
                        # 2. Stats CSV
                        # Drop the 'Deltas' list column for CSV export
                        stats_export = mmp_stats_df.drop(columns=['Deltas'])
                        csv_stats = stats_export.to_csv(index=False)
                        c2.download_button(
                            "📥 Download Transform Stats (.csv)",
                            csv_stats,
                            "mmp_stats.csv",
                            "text/csv"
                        )
                        
                    else:
                        st.warning("Analysis complete, but no transforms met the minimum occurrence criteria.")
                        if pairs_df is not None:
                            st.write(f"Total raw pairs generated: {len(pairs_df)}")
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main()
