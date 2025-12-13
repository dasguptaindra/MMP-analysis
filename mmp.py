import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMMPA import FragmentMol as RDKitFragmentMol
from operator import itemgetter
from itertools import combinations
import mols2grid
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit.Chem.Draw import rdMolDraw2D
import io
import base64

# Embedded definitions from scaffold_finder.py and useful_rdkit_utils.py

def FragmentMol(mol, maxCuts=2, minCuts=1, minFragSize=1, maxFragSize=None,
                pattern="[!$(*#*)&!D1]-!@[!$(*#*)&!D1]", resultsAsMols=True,
                functionalGroups=[], sanitize=True):
    """
    Generate fragments by breaking bonds matching a SMARTS pattern.
    This is a wrapper around rdkit.Chem.rdMMPA.FragmentMol.
    """
    if maxFragSize is None:
        maxFragSize = mol.GetNumAtoms()

    res = RDKitFragmentMol(mol, maxCuts=maxCuts, minCuts=minCuts,
                                 pattern=pattern, resultsAsMols=resultsAsMols,
                                 functionalGroups=functionalGroups, sanitize=sanitize)
    return res

def get_largest_fragment(mol):
    """
    Returns the largest fragment of a molecule.
    """
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True)
    if not frags:
        return None
    largest_frag = max(frags, key=lambda m: m.GetNumAtoms())
    return largest_frag

# Other helper functions
def remove_map_nums(mol):
    """
    Remove atom map numbers from a molecule
    """
    for atm in mol.GetAtoms():
        atm.SetAtomMapNum(0)

def sort_fragments(mol):
    """
    Transform a molecule with multiple fragments into a list of molecules that is sorted by number of atoms
    from largest to smallest
    """
    frag_list = list(Chem.GetMolFrags(mol, asMols=True))
    [remove_map_nums(x) for x in frag_list]
    frag_num_atoms_list = [(x.GetNumAtoms(), x) for x in frag_list]
    frag_num_atoms_list.sort(key=itemgetter(0), reverse=True)
    return [x[1] for x in frag_num_atoms_list]

def rxn_to_base64_image(rxn):
    """
    Convert an RDKit reaction to an image in base64 format.
    """
    try:
        drawer = rdMolDraw2D.MolDraw2DCairo(300, 150)
        drawer.DrawReaction(rxn)
        drawer.FinishDrawing()
        bio = io.BytesIO(drawer.GetDrawingText())
        im_text64 = base64.b64encode(bio.getvalue()).decode('utf8')
        img_str = f"<img src='data:image/png;base64, {im_text64}'/>"
        return img_str
    except Exception as e:
        st.warning(f"Could not draw reaction: {e}")
        return "Error rendering reaction"

def stripplot_base64_image(dist):
    """
    Plot a distribution as a seaborn stripplot and save the resulting image as a base64 string.
    """
    sns.set(rc={'figure.figsize': (3, 1)})
    sns.set_style('whitegrid')
    fig, ax = plt.subplots()
    sns.stripplot(x=dist, ax=ax)
    ax.axvline(0, ls="--", c="red")
    ax.set_xlim(min(dist)-0.5 if len(dist) > 0 else -1, max(dist)+0.5 if len(dist) > 0 else 1)
    ax.set_yticks([]) # Remove y-axis ticks
    ax.set_xlabel('') # Remove x-axis label
    s = io.BytesIO()
    plt.savefig(s, format='png', bbox_inches="tight", dpi=100)
    plt.close(fig) # Close the figure to free memory
    s = base64.b64encode(s.getvalue()).decode("utf-8").replace("\n", "")
    return '<img align="left" src="data:image/png;base64,%s">' % s

def find_examples(delta_df, query_idx):
    example_list = []
    # Filter delta_df for the given query_idx and sort by Delta
    filtered_df = delta_df.query("idx == @query_idx").sort_values("Delta", ascending=False)

    for _, row in filtered_df.iterrows():
        smi_1, name_1, pIC50_1 = row.SMILES_1, row.Name_1, row.pIC50_1
        smi_2, name_2, pIC50_2 = row.SMILES_2, row.Name_2, row.pIC50_2
        # Ensure consistent ordering for pairs if needed, but the original notebook sorted SMILES for delta_df creation
        # Here, we just append both as separate entries for mols2grid display
        example_list.append([smi_1, name_1, pIC50_1])
        example_list.append([smi_2, name_2, pIC50_2])

    example_df = pd.DataFrame(example_list, columns=["SMILES", "Name", "pIC50"])
    return example_df

@st.cache_data
def load_and_process_data(file_path):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"File not found: {file_path}. Please ensure it's in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    df['mol'] = df.SMILES.apply(Chem.MolFromSmiles)
    df.mol = df.mol.apply(get_largest_fragment)

    row_list = []
    # The original notebook uses tqdm.pandas(), but tqdm is for progress bars in terminals/notebooks.
    # For Streamlit, we might use st.progress or just omit it for simplicity as the function will be cached.
    for smiles, name, pIC50, mol in df.values:
        if mol is None: # Skip rows where MolFromSmiles or get_largest_fragment failed
            continue
        frag_list = FragmentMol(mol, maxCuts=1)
        for _, frag_mol_tuple in frag_list:
            pair_list = sort_fragments(frag_mol_tuple)
            if len(pair_list) == 2:
                # Ensure Core and R_group are correctly assigned from the two fragments
                # The original logic implicitly assigned by order, assuming pair_list[0] is Core, pair_list[1] is R_group
                # However, the notebook output implies pair_list[0] is Core and pair_list[1] is R_group based on the example
                # For robustness, we should explicitly check the fragments' roles if possible, but following the original flow:
                # Core is the fragment that retains more of the original structure or is simply the larger one by convention.
                # Given `sort_fragments` sorts by size, pair_list[0] will be the larger fragment.
                tmp_list = [smiles, Chem.MolToSmiles(pair_list[0]), Chem.MolToSmiles(pair_list[1]), name, pIC50]
                row_list.append(tmp_list)

    row_df = pd.DataFrame(row_list, columns=["SMILES", "Core", "R_group", "Name", "pIC50"])

    delta_list = []
    # Iterating over grouped dataframes. For performance in Streamlit, ensure this is optimized if large.
    for k, v in row_df.groupby("Core"):
        if len(v) > 2:
            for a, b in combinations(range(0, len(v)), 2):
                reagent_a = v.iloc[a]
                reagent_b = v.iloc[b]
                if reagent_a.SMILES == reagent_b.SMILES:
                    continue
                # Sort reagents based on SMILES to ensure canonical ordering of pairs
                reagent_a, reagent_b = sorted([reagent_a, reagent_b], key=lambda x: x.SMILES)
                delta = reagent_b.pIC50 - reagent_a.pIC50
                delta_list.append(list(reagent_a.values) + list(reagent_b.values) +
                                  [f"{reagent_a.R_group.replace('*','*-')}>>{reagent_b.R_group.replace('*','*-')}", delta])

    cols = ["SMILES_1", "Core_1", "R_group_1", "Name_1", "pIC50_1",
            "SMILES_2", "Core_2", "R_group_2", "Name_2", "pIC50_2",
            "Transform", "Delta"]
    delta_df = pd.DataFrame(delta_list, columns=cols)

    min_transform_occurrence = 5 # As defined in the notebook
    mmp_list = []
    for k, v in delta_df.groupby("Transform"):
        if len(v) > min_transform_occurrence:
            mmp_list.append([k, len(v), v.Delta.values])

    mmp_df = pd.DataFrame(mmp_list, columns=["Transform", "Count", "Deltas"])
    mmp_df['idx'] = range(0, len(mmp_df))
    mmp_df['mean_delta'] = [x.mean() for x in mmp_df.Deltas]
    mmp_df['rxn_mol'] = mmp_df.Transform.apply(lambda x: AllChem.ReactionFromSmarts(x.replace('*-','*'), useSmiles=True) if x is not None else None)
    
    # Create a dictionary to link Transform to idx for delta_df
    transform_dict = dict([(a,b) for a,b in mmp_df[["Transform","idx"]].values])
    delta_df['idx'] = [transform_dict.get(x) for x in delta_df.Transform]

    # Generate base64 images for display, this part is not cached with @st.cache_data
    # as it relies on Streamlit rendering directly, but can be pre-computed if needed.
    # For this task, it's better to calculate these within the main app logic or only when needed.
    # However, for the display table, pre-calculating makes sense.

    return df, row_df, delta_df, mmp_df

def main():
    st.title("Matched Molecular Pairs Analysis")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    data_file_path = "MMP_9_MMP.csv" # Default local file
    if uploaded_file is not None:
        # Read the uploaded file into a temporary path or directly as pandas dataframe
        # For st.cache_data to work, if file is uploaded, hash changes, so data will be reloaded.
        # Let's save the uploaded file to a temporary location to make it consistent with file_path argument.
        with open(data_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("File uploaded successfully.")

    st.write(f"Loading data from: {data_file_path}")
    df_orig, row_df, delta_df, mmp_df = load_and_process_data(data_file_path)

    st.header("MMP Transformations Overview")

    # Calculate and store base64 images only once
    mmp_df['MMP Transform'] = mmp_df.rxn_mol.apply(rxn_to_base64_image)
    mmp_df['Delta Distribution'] = mmp_df.Deltas.apply(stripplot_base64_image)

    # UI for sorting
    sort_by_options = {"Mean Delta (Ascending)": True, "Mean Delta (Descending)": False, "Count (Ascending)": True, "Count (Descending)": False}
    selected_sort = st.selectbox("Sort transformations by:", list(sort_by_options.keys()))
    
    if "Mean Delta" in selected_sort:
        mmp_df_sorted = mmp_df.sort_values("mean_delta", ascending=sort_by_options[selected_sort])
        display_cols = ['MMP Transform', 'Count', "mean_delta", "Delta Distribution"]
    elif "Count" in selected_sort:
        mmp_df_sorted = mmp_df.sort_values("Count", ascending=sort_by_options[selected_sort])
        display_cols = ['MMP Transform', 'Count', "mean_delta", "Delta Distribution"]
    
    st.write("### Top MMPs")
    # Display the HTML table
    st.markdown(mmp_df_sorted[display_cols].round(2).to_html(escape=False), unsafe_allow_html=True)

    st.header("Explore Specific Transformations")
    # User selects a transformation by its index or description
    transform_options = mmp_df_sorted.apply(lambda row: f"ID {row.idx}: {row.Transform} (Mean Delta: {row.mean_delta:.2f}, Count: {row.Count})", axis=1).tolist()
    selected_transform_str = st.selectbox("Select a transformation to view examples:", transform_options)

    # Extract query_idx from the selected string
    if selected_transform_str:
        query_idx_str = selected_transform_str.split(':')[0].replace('ID ', '')
        query_idx = int(query_idx_str)

        st.subheader(f"Examples for Transformation ID {query_idx}")
        
        # Get the actual transform string for display
        actual_transform = mmp_df.loc[mmp_df['idx'] == query_idx, 'Transform'].iloc[0]
        st.write(f"**Transformation:** {actual_transform}")
        st.write(f"**Mean \u0394pIC50:** {mmp_df.loc[mmp_df['idx'] == query_idx, 'mean_delta'].iloc[0]:.2f}")
        st.write(f"**Occurrences:** {mmp_df.loc[mmp_df['idx'] == query_idx, 'Count'].iloc[0]}")

        example_df = find_examples(delta_df, query_idx)
        # Rename columns for mols2grid, assuming SMILES is in 'Name' after find_examples for current data
        # and actual SMILES are in 'SMILES' which need to become ChEMBL_ID for display.
        # This is a bit convoluted due to original notebook's column usage in find_examples output
        # Let's adjust find_examples to return original SMILES in 'SMILES' and Name in 'Name' for clarity
        # Re-evaluating find_examples' output. It creates ['SMILES', 'Name', 'pIC50'].
        # The mol2grid call `smiles_col="SMILES"` implies 'SMILES' contains SMILES strings.
        # However, the previous analysis `print(example_df['SMILES'].head(3).tolist())` showed 'CHEMBL...' IDs.
        # This means the 'SMILES' column from `find_examples` is actually the ChEMBL ID, and 'Name' is the SMILES.
        # So the column swap is correct here.

        example_df_fixed = example_df.rename(columns={"SMILES": "ChEMBL_ID", "Name": "SMILES"})

        if not example_df_fixed.empty:
            st.write(mols2grid.display(
                example_df_fixed,
                smiles_col="SMILES",
                n_cols=4,
                template='static',
                prerender=True,
                size=(200, 200),
                subset=["img", "ChEMBL_ID", "pIC50"],
                transform={"pIC50": lambda x: f"{x:.2f}"}
            )._repr_html_(), unsafe_allow_html=True)
        else:
            st.write("No examples found for this transformation.")


if __name__ == '__main__':
    main()
