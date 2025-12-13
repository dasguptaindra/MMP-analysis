import streamlit as st
import sys
from rdkit import Chem
from rdkit.Chem.rdMMPA import FragmentMol
from rdkit.Chem.rdRGroupDecomposition import RGroupDecompose
import itertools
import useful_rdkit_utils as uru
import pandas as pd
# Using standard tqdm in streamlit usually prints to console, 
# for UI progress bars we'd use st.progress, but to keep "exact implementation" 
# we'll use tqdm which works in logs, or simply process directly.
from tqdm import tqdm 
import numpy as np
import io

# ---------------------------------------------------------
# Core Functions (Exactly as provided in scaffold_finder.py)
# ---------------------------------------------------------

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
    
    # In Streamlit, we can show a status text instead of console tqdm
    progress_text = "Generating fragments..."
    my_bar = st.progress(0, text=progress_text)
    
    total_mols = len(df_in)
    
    for i, (smiles, name, mol) in enumerate(df_in[["SMILES", "Name", "mol"]].values):
        tmp_df = generate_fragments(mol).copy()
        tmp_df['Name'] = name
        tmp_df['SMILES'] = smiles
        df_list.append(tmp_df)
        
        # Update progress bar
        percent_complete = int((i + 1) / total_mols * 100)
        my_bar.progress(percent_complete, text=f"Processing molecule {i+1}/{total_mols}")
        
    my_bar.empty() # Clear progress bar when done

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


# ---------------------------------------------------------
# Streamlit App Logic
# ---------------------------------------------------------

def main():
    st.set_page_config(page_title="Scaffold Finder", layout="wide")
    st.title("Scaffold Stats Generator")
    st.markdown("""
    This app replicates the `scaffold_finder.py` logic. 
    Upload SMILES files (CSV format: `SMILES, Name, pIC50`) to generate scaffold statistics.
    """)

    # File uploader acts as the 'glob' replacement
    uploaded_files = st.file_uploader("Upload SMILES files (.smi or .csv)", accept_multiple_files=True)

    if uploaded_files:
        if st.button("Run Scaffold Analysis"):
            out_list = []
            
            # Processing container
            for uploaded_file in uploaded_files:
                filename = uploaded_file.name
                st.subheader(f"Processing: {filename}")
                
                try:
                    # Reset file pointer to beginning
                    uploaded_file.seek(0)
                    
                    # Exact same reading logic: names=["SMILES", "Name", "pIC50"]
                    input_df = pd.read_csv(uploaded_file, names=["SMILES", "Name", "pIC50"])
                    
                    # Molecular conversion
                    st.text("Converting SMILES to Mols...")
                    input_df['mol'] = input_df.SMILES.apply(Chem.MolFromSmiles)
                    # Remove invalid molecules if any (optional, but good practice for apps)
                    if input_df['mol'].isnull().any():
                        st.warning(f"Warning: {input_df['mol'].isnull().sum()} invalid SMILES found and dropped.")
                        input_df = input_df.dropna(subset=['mol'])
                    
                    input_df['SMILES'] = input_df.mol.apply(Chem.MolToSmiles)
                    
                    # Core Logic
                    mol_df, scaffold_df = find_scaffolds(input_df)
                    
                    if scaffold_df.empty:
                         st.error(f"Could not find any valid scaffolds for {filename}")
                         continue

                    scaffold_1 = scaffold_df.Scaffold.values[0]
                    scaffold_list, scaffold_mol_df = get_molecules_with_scaffold(scaffold_1, mol_df, input_df)
                    
                    ic50_list = scaffold_mol_df.pIC50
                    
                    if len(scaffold_list):
                        # Calculate stats
                        stats_count = len(scaffold_mol_df)
                        stats_range = max(ic50_list) - min(ic50_list)
                        stats_std = np.std(ic50_list)
                        
                        # Display intermediate result exactly as printed in original script
                        st.write(f"**Top Scaffold:** `{scaffold_list[0]}`")
                        st.write(f"Count: {stats_count}, Range: {stats_range:.4f}, Std: {stats_std:.4f}")
                        
                        out_list.append([filename, scaffold_list[0], stats_count, stats_range, stats_std])
                    else:
                        st.error(f"Could not find a scaffold for {filename}")

                except Exception as e:
                    st.error(f"Error processing {filename}: {str(e)}")

            # Final Results
            if out_list:
                st.divider()
                st.header("Results (scaffold_stats.csv)")
                out_df = pd.DataFrame(out_list, columns=["Filename", "Scaffold", "Count", "Range", "Std"])
                st.dataframe(out_df)
                
                # CSV Download
                csv = out_df.to_csv(index=False)
                st.download_button(
                    label="Download scaffold_stats.csv",
                    data=csv,
                    file_name="scaffold_stats.csv",
                    mime="text/csv",
                )

if __name__ == "__main__":
    main()
