import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdMMPA import FragmentMol
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.SaltRemover import SaltRemover
from operator import itemgetter
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from rdkit.Chem.Draw import rdMolDraw2D
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(
    page_title="MMP Analysis Tool",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Matched Molecular Pair (MMP) Analysis")

# -----------------------------
# Helper Functions
# -----------------------------

def get_largest_fragment(mol):
    """Standardizes the molecule by removing salts and keeping the largest organic fragment."""
    if mol is None:
        return None
    try:
        remover = SaltRemover()
        mol = remover.StripMol(mol, dontRemoveEverything=True)
        frags = list(Chem.GetMolFrags(mol, asMols=True))
        if not frags:
            return mol
        return max(frags, key=lambda m: m.GetNumAtoms())
    except Exception:
        return mol

def remove_map_nums(mol):
    for atm in mol.GetAtoms():
        atm.SetAtomMapNum(0)

def sort_fragments(mol):
    """Splits the fragmented molecule into constituent parts."""
    if mol is None:
        return []
    try:
        frag_list = list(Chem.GetMolFrags(mol, asMols=True))
        for x in frag_list:
            remove_map_nums(x)
        frag_num_atoms_list = [(x.GetNumAtoms(), x) for x in frag_list]
        frag_num_atoms_list.sort(key=itemgetter(0), reverse=True)
        return [x[1] for x in frag_num_atoms_list]
    except Exception:
        return []

def rxn_to_base64_image(rxn):
    """Generates a base64 encoded SVG of the reaction."""
    try:
        drawer = rdMolDraw2D.MolDraw2DSVG(400, 200)
        drawer.DrawReaction(rxn)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        b64 = base64.b64encode(svg.encode()).decode()
        return f'<img src="data:image/svg+xml;base64,{b64}"/>'
    except:
        return ""

def mol_to_base64_image(mol, size=(200, 150)):
    """Converts RDKit molecule to base64 image."""
    try:
        img = Draw.MolToImage(mol, size=size)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        return f'<img src="data:image/png;base64,{b64}"/>'
    except:
        return ""

def create_distribution_plot(deltas, transform_name):
    """Creates an interactive distribution plot using Plotly."""
    fig = go.Figure()
    
    # Add stripplot points
    fig.add_trace(go.Box(
        y=deltas,
        name=transform_name[:30] + "..." if len(transform_name) > 30 else transform_name,
        boxpoints='all',
        jitter=0.3,
        pointpos=-1.8,
        marker=dict(size=8, color='#1f77b4', opacity=0.7),
        line=dict(width=2, color='#1f77b4')
    ))
    
    # Add mean line
    mean_val = np.mean(deltas)
    fig.add_hline(y=mean_val, line_dash="dash", line_color="red", 
                  annotation_text=f"Mean: {mean_val:.2f}")
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    
    fig.update_layout(
        title=f"Distribution of ΔpIC50 for {transform_name[:40]}...",
        yaxis_title="ΔpIC50",
        height=300,
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_summary_plots(mmp_df):
    """Creates summary plots for the analysis."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Transform Frequency Distribution', 
                       'Mean ΔpIC50 vs Frequency',
                       'ΔpIC50 Distribution Across All Transforms',
                       'Top 10 Most Frequent Transforms'),
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )
    
    # 1. Transform Frequency Distribution
    fig.add_trace(
        go.Histogram(x=mmp_df['Count'], nbinsx=20, name='Frequency Distribution',
                    marker_color='#636efa'),
        row=1, col=1
    )
    fig.update_xaxes(title_text="Number of Pairs", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    
    # 2. Mean ΔpIC50 vs Frequency
    fig.add_trace(
        go.Scatter(x=mmp_df['Count'], y=mmp_df['mean_delta'],
                  mode='markers', name='Transforms',
                  marker=dict(size=8, color=mmp_df['mean_delta'], 
                            colorscale='RdBu', showscale=True,
                            colorbar=dict(title="Mean ΔpIC50")),
                  text=mmp_df['Transform'].str[:30]),
        row=1, col=2
    )
    fig.update_xaxes(title_text="Number of Pairs", row=1, col=2)
    fig.update_yaxes(title_text="Mean ΔpIC50", row=1, col=2)
    
    # 3. Overall ΔpIC50 Distribution
    all_deltas = np.concatenate(mmp_df['Deltas'].values)
    fig.add_trace(
        go.Histogram(x=all_deltas, nbinsx=30, name='ΔpIC50 Distribution',
                    marker_color='#00cc96'),
        row=2, col=1
    )
    fig.update_xaxes(title_text="ΔpIC50", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    
    # 4. Top 10 Transforms
    top10 = mmp_df.nlargest(10, 'Count')
    fig.add_trace(
        go.Bar(x=top10['Transform'].str[:20] + "...", 
              y=top10['Count'], name='Top 10',
              marker_color='#ff7f0e'),
        row=2, col=2
    )
    fig.update_xaxes(title_text="Transform", row=2, col=2, tickangle=45)
    fig.update_yaxes(title_text="Count", row=2, col=2)
    
    fig.update_layout(height=700, showlegend=False, title_text="MMP Analysis Summary")
    return fig

def to_excel_bytes(df_dict, sheet_names):
    """Convert multiple dataframes to Excel bytes."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for df, sheet_name in zip(df_dict.values(), sheet_names):
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    output.seek(0)
    return output

# -----------------------------
# Sidebar & Input
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    
    st.subheader("MMP Settings")
    min_transform_occurrence = st.slider(
        "Minimum MMP Occurrence", 2, 50, 3,
        help="Only show transformations that appear at least this many times in the dataset."
    )
    
    max_delta_display = st.slider(
        "Maximum ΔpIC50 to Display", 0.0, 10.0, 5.0, 0.1,
        help="Filter transformations by absolute ΔpIC50 value"
    )
    
    st.subheader("Data Input")
    uploaded_file = st.file_uploader(
        "Upload CSV (must contain SMILES, Name, pIC50)", type=["csv"]
    )
    
    st.subheader("Visualization")
    show_details = st.checkbox("Show Detailed Compound Examples", value=True)
    show_summary = st.checkbox("Show Summary Statistics", value=True)

# -----------------------------
# Main Logic
# -----------------------------
if uploaded_file:
    # 1. Load Data
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()
    
    # 2. Column Mapping
    st.subheader("1. 📊 Data Mapping")
    cols = df.columns.tolist()
    
    c1, c2, c3 = st.columns(3)
    smiles_col = c1.selectbox("Select SMILES column", cols, 
                             index=cols.index("SMILES") if "SMILES" in cols else 0)
    name_col = c2.selectbox("Select ID/Name column", cols, 
                           index=cols.index("Name") if "Name" in cols else 0)
    pic50_col = c3.selectbox("Select Activity (pIC50) column", cols, 
                            index=cols.index("pIC50") if "pIC50" in cols else 0)

    # Validate data
    initial_count = len(df)
    df = df.dropna(subset=[smiles_col, pic50_col])
    df[pic50_col] = pd.to_numeric(df[pic50_col], errors='coerce')
    df = df.dropna(subset=[pic50_col])
    valid_count = len(df)
    
    with st.expander("📈 Data Overview", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Compounds", initial_count)
        col2.metric("Valid Compounds", valid_count)
        col3.metric("Data Loss", f"{((initial_count-valid_count)/initial_count*100):.1f}%")
        col4.metric("pIC50 Range", 
                   f"{df[pic50_col].min():.2f} - {df[pic50_col].max():.2f}")
        
        # Show data preview
        st.dataframe(df[[smiles_col, name_col, pic50_col]].head(), use_container_width=True)

    # 3. Processing
    if st.button("🚀 Run MMP Analysis", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Track statistics
        stats = {
            'total_compounds': valid_count,
            'valid_mols': 0,
            'fragments_generated': 0,
            'pairs_generated': 0,
            'transforms_found': 0,
            'transforms_filtered': 0
        }
        
        with st.spinner("Processing..."):
            status_text.text("Step 1/4: Processing molecules...")
            progress_bar.progress(10)
            
            # Prepare Mols
            df["mol"] = df[smiles_col].apply(Chem.MolFromSmiles)
            df_valid = df.dropna(subset=["mol"])
            stats['valid_mols'] = len(df_valid)
            
            # Apply standardization
            df_valid["mol"] = df_valid["mol"].apply(get_largest_fragment)
            
            status_text.text("Step 2/4: Fragmenting molecules...")
            progress_bar.progress(30)
            
            # Fragmentation
            row_list = []
            for idx, row in df_valid.iterrows():
                mol = row["mol"]
                smiles = row[smiles_col]
                name = row[name_col]
                pIC50 = row[pic50_col]
                
                if mol:
                    try:
                        frags = FragmentMol(mol, maxCuts=1, resultsAsMols=True)
                    except Exception:
                        continue

                    for frag_item in frags:
                        if isinstance(frag_item, tuple):
                            frag_mol = frag_item[1]
                        else:
                            frag_mol = frag_item
                        
                        pair = sort_fragments(frag_mol)
                        stats['fragments_generated'] += 1
                        
                        if len(pair) == 2:
                            core_smi = Chem.MolToSmiles(pair[0])
                            r_smi = Chem.MolToSmiles(pair[1])
                            
                            row_list.append([
                                smiles, core_smi, r_smi, name, pIC50,
                                mol_to_base64_image(pair[0], (150, 100)),
                                mol_to_base64_image(pair[1], (100, 75))
                            ])

            row_df = pd.DataFrame(
                row_list,
                columns=["SMILES", "Core", "R_group", "Name", "pIC50", 
                        "Core_Image", "R_Group_Image"]
            )
            
            stats['pairs_generated'] = len(row_list)
            
            if row_df.empty:
                st.error("No valid fragments generated.")
                st.stop()
            
            status_text.text("Step 3/4: Calculating ΔpIC50 values...")
            progress_bar.progress(60)
            
            # Delta Calculation
            delta_rows = []
            grouped = row_df.groupby("Core")
            
            for core, group in grouped:
                if len(group) >= 2:
                    group_indices = range(len(group))
                    for i, j in combinations(group_indices, 2):
                        ra = group.iloc[i]
                        rb = group.iloc[j]
                        
                        if ra.R_group == rb.R_group:
                            continue
                        
                        mols_sorted = sorted([ra, rb], key=lambda x: x.R_group)
                        r1, r2 = mols_sorted[0], mols_sorted[1]
                        
                        transform_str = f"{r1.R_group.replace('*','*-')}>>{r2.R_group.replace('*','*-')}"
                        delta_val = r2.pIC50 - r1.pIC50
                        
                        delta_rows.append([
                            r1.SMILES, r1.Core, r1.R_group, r1.Name, r1.pIC50, r1.Core_Image,
                            r2.SMILES, r2.Core, r2.R_group, r2.Name, r2.pIC50, r2.Core_Image,
                            transform_str, delta_val
                        ])

            delta_df = pd.DataFrame(delta_rows, columns=[
                "SMILES_1", "Core_1", "R_group_1", "Name_1", "pIC50_1", "Core_Image_1",
                "SMILES_2", "Core_2", "R_group_2", "Name_2", "pIC50_2", "Core_Image_2",
                "Transform", "Delta"
            ])
            
            stats['transforms_found'] = len(delta_df)
            
            status_text.text("Step 4/4: Aggregating results...")
            progress_bar.progress(90)
            
            # Aggregation
            mmp_rows = []
            for transform, v in delta_df.groupby("Transform"):
                if len(v) >= min_transform_occurrence:
                    deltas = v.Delta.values
                    if np.max(np.abs(deltas)) <= max_delta_display:
                        mmp_rows.append([transform, len(v), deltas])
            
            mmp_df = pd.DataFrame(mmp_rows, columns=["Transform", "Count", "Deltas"])
            stats['transforms_filtered'] = len(mmp_df)
            
            if not mmp_df.empty:
                mmp_df["mean_delta"] = mmp_df["Deltas"].apply(np.mean)
                mmp_df["std_delta"] = mmp_df["Deltas"].apply(np.std)
                mmp_df["min_delta"] = mmp_df["Deltas"].apply(np.min)
                mmp_df["max_delta"] = mmp_df["Deltas"].apply(np.max)
                
                # Create Reaction Objects
                mmp_df["rxn"] = mmp_df["Transform"].apply(
                    lambda x: AllChem.ReactionFromSmarts(x.replace("*-", "*"), useSmiles=True)
                )
                mmp_df["MMP Transform"] = mmp_df["rxn"].apply(rxn_to_base64_image)
                
            progress_bar.progress(100)
            status_text.text("Analysis complete!")
            
        # 4. Display Results
        st.success(f"✅ Analysis completed successfully!")
        
        # Summary Statistics
        with st.expander("📊 Summary Statistics", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Compounds", stats['total_compounds'])
            col2.metric("Valid Molecules", stats['valid_mols'])
            col3.metric("Fragment Pairs", stats['pairs_generated'])
            col4.metric("MMP Pairs Found", stats['transforms_found'])
            
            col5, col6, col7, col8 = st.columns(4)
            col5.metric("Transform Filter", f"≥{min_transform_occurrence}")
            col6.metric("ΔpIC50 Filter", f"≤{max_delta_display}")
            col7.metric("Transforms Retained", stats['transforms_filtered'])
            col8.metric("Data Coverage", 
                       f"{(stats['transforms_filtered']/max(1, stats['transforms_found'])*100):.1f}%")
            
            # Create summary table
            summary_df = pd.DataFrame({
                'Metric': list(stats.keys()),
                'Value': list(stats.values())
            })
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Show summary plots
        if show_summary and not mmp_df.empty:
            st.subheader("📈 Analysis Summary Plots")
            summary_fig = create_summary_plots(mmp_df)
            st.plotly_chart(summary_fig, use_container_width=True)
        
        # Detailed Results
        if not mmp_df.empty:
            st.subheader(f"🔍 Detailed MMP Results ({len(mmp_df)} transforms)")
            
            # Sort options
            sort_by = st.selectbox("Sort by:", 
                                  ["Count (Descending)", "Mean ΔpIC50 (Abs)", "Mean ΔpIC50 (Asc)", "Mean ΔpIC50 (Desc)"])
            
            if sort_by == "Count (Descending)":
                mmp_df = mmp_df.sort_values("Count", ascending=False)
            elif sort_by == "Mean ΔpIC50 (Abs)":
                mmp_df = mmp_df.iloc[np.argsort(np.abs(mmp_df["mean_delta"].values))[::-1]]
            elif sort_by == "Mean ΔpIC50 (Asc)":
                mmp_df = mmp_df.sort_values("mean_delta", ascending=True)
            else:
                mmp_df = mmp_df.sort_values("mean_delta", ascending=False)
            
            # Display table with pagination
            page_size = st.slider("Rows per page:", 5, 50, 10)
            total_pages = len(mmp_df) // page_size + (1 if len(mmp_df) % page_size else 0)
            
            if total_pages > 1:
                page_num = st.number_input("Page:", 1, total_pages, 1)
                start_idx = (page_num - 1) * page_size
                end_idx = min(start_idx + page_size, len(mmp_df))
                display_df = mmp_df.iloc[start_idx:end_idx].copy()
                st.caption(f"Showing rows {start_idx+1} to {end_idx} of {len(mmp_df)}")
            else:
                display_df = mmp_df.copy()
            
            # Create interactive display
            for idx, row in display_df.iterrows():
                with st.expander(f"{row['Transform']} (Count: {row['Count']}, Mean Δ: {row['mean_delta']:.2f})", 
                               expanded=False):
                    col1, col2, col3 = st.columns([2, 3, 2])
                    
                    with col1:
                        st.markdown("**Transform:**")
                        st.markdown(row['MMP Transform'], unsafe_allow_html=True)
                        st.markdown(f"**Occurrences:** {row['Count']}")
                        st.markdown(f"**Mean ΔpIC50:** {row['mean_delta']:.2f} ± {row['std_delta']:.2f}")
                        st.markdown(f"**Range:** [{row['min_delta']:.2f}, {row['max_delta']:.2f}]")
                    
                    with col2:
                        fig = create_distribution_plot(row['Deltas'], row['Transform'])
                        st.plotly_chart(fig, use_container_width=True, use_container_height=True)
                    
                    with col3:
                        # Show example compounds for this transform
                        st.markdown("**Example Compounds:**")
                        examples = delta_df[delta_df['Transform'] == row['Transform']].head(3)
                        for ex_idx, ex_row in examples.iterrows():
                            with st.container():
                                st.markdown(f"**Pair {ex_idx+1}:** Δ = {ex_row['Delta']:.2f}")
                                cols_ex = st.columns(2)
                                with cols_ex[0]:
                                    st.markdown(f"*{ex_row['Name_1']}*: {ex_row['pIC50_1']:.2f}")
                                    st.markdown(ex_row['Core_Image_1'], unsafe_allow_html=True)
                                with cols_ex[1]:
                                    st.markdown(f"*{ex_row['Name_2']}*: {ex_row['pIC50_2']:.2f}")
                                    st.markdown(ex_row['Core_Image_2'], unsafe_allow_html=True)
                                st.divider()
            
            # Download Section
            st.subheader("💾 Download Results")
            
            # Prepare data for download
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create detailed results dataframe
            detailed_results = []
            for _, mmp_row in mmp_df.iterrows():
                transform_deltas = delta_df[delta_df['Transform'] == mmp_row['Transform']]
                for _, pair_row in transform_deltas.iterrows():
                    detailed_results.append({
                        'Transform': mmp_row['Transform'],
                        'Transform_Count': mmp_row['Count'],
                        'Transform_Mean_Delta': mmp_row['mean_delta'],
                        'Compound1_Name': pair_row['Name_1'],
                        'Compound1_SMILES': pair_row['SMILES_1'],
                        'Compound1_pIC50': pair_row['pIC50_1'],
                        'Compound2_Name': pair_row['Name_2'],
                        'Compound2_SMILES': pair_row['SMILES_2'],
                        'Compound2_pIC50': pair_row['pIC50_2'],
                        'Delta_pIC50': pair_row['Delta'],
                        'Core_SMILES': pair_row['Core_1']
                    })
            
            detailed_df = pd.DataFrame(detailed_results)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.download_button(
                    label="📥 Download Summary (CSV)",
                    data=mmp_df.drop(columns=['rxn', 'MMP Transform', 'Deltas']).to_csv(index=False),
                    file_name=f"mmp_summary_{timestamp}.csv",
                    mime="text/csv"
                )
            
            with col2:
                st.download_button(
                    label="📥 Download Detailed Pairs (CSV)",
                    data=detailed_df.to_csv(index=False),
                    file_name=f"mmp_detailed_pairs_{timestamp}.csv",
                    mime="text/csv"
                )
            
            with col3:
                # Create Excel with multiple sheets
                excel_data = to_excel_bytes(
                    {
                        'Summary': mmp_df.drop(columns=['rxn', 'MMP Transform', 'Deltas']),
                        'Detailed_Pairs': detailed_df,
                        'Raw_Fragments': row_df.drop(columns=['Core_Image', 'R_Group_Image']),
                        'Original_Data': df_valid[[smiles_col, name_col, pic50_col]]
                    },
                    ['Summary', 'Detailed_Pairs', 'Raw_Fragments', 'Original_Data']
                )
                
                st.download_button(
                    label="📥 Download Full Report (Excel)",
                    data=excel_data,
                    file_name=f"mmp_full_report_{timestamp}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            # Report generation
            with st.expander("📋 Generate Analysis Report"):
                report_text = f"""
                # MMP Analysis Report
                **Generated:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                **Dataset:** {uploaded_file.name}
                **Total Compounds:** {stats['total_compounds']}
                **Valid Compounds:** {stats['valid_mols']}
                
                ## Analysis Parameters
                - Minimum Transform Occurrence: {min_transform_occurrence}
                - Maximum ΔpIC50 Display: {max_delta_display}
                
                ## Results Summary
                - Fragment Pairs Generated: {stats['pairs_generated']}
                - Total MMP Pairs Found: {stats['transforms_found']}
                - Transforms Retained after Filtering: {stats['transforms_filtered']}
                
                ## Top Transformations
                """
                
                # Add top transformations
                top5 = mmp_df.head(5)
                for i, (_, row) in enumerate(top5.iterrows(), 1):
                    report_text += f"\n{i}. **{row['Transform']}**\n"
                    report_text += f"   - Count: {row['Count']}\n"
                    report_text += f"   - Mean ΔpIC50: {row['mean_delta']:.2f} ± {row['std_delta']:.2f}\n"
                    report_text += f"   - Range: [{row['min_delta']:.2f}, {row['max_delta']:.2f}]\n"
                
                st.download_button(
                    label="📄 Download Text Report",
                    data=report_text,
                    file_name=f"mmp_text_report_{timestamp}.txt",
                    mime="text/plain"
                )
        
        else:
            st.warning(f"No transformations found with occurrence ≥ {min_transform_occurrence} and |ΔpIC50| ≤ {max_delta_display}")
            
            # Still provide download for raw fragments
            if not row_df.empty:
                st.download_button(
                    label="📥 Download Raw Fragments",
                    data=row_df.drop(columns=['Core_Image', 'R_Group_Image']).to_csv(index=False),
                    file_name="mmp_raw_fragments.csv",
                    mime="text/csv"
                )

else:
    st.info("👈 Upload a CSV file to start analysis.")
    
    # Show example format
    with st.expander("📋 Expected CSV Format"):
        example_data = {
            'SMILES': ['CC(=O)Oc1ccccc1C(=O)O', 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O', 'C1=CC=C(C=C1)C=O'],
            'Name': ['Aspirin', 'Ibuprofen', 'Benzaldehyde'],
            'pIC50': [4.5, 5.2, 3.8]
        }
        example_df = pd.DataFrame(example_data)
        st.dataframe(example_df, use_container_width=True)
        st.caption("Required columns: SMILES, Name (or ID), pIC50 (or other activity metric)")
