import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMMPA, AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from tqdm import tqdm
from itertools import combinations
import base64
import io
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------

def get_largest_fragment(mol):
    """
    Returns the largest fragment of a molecule (filters out salts/solvents).
    """
    if mol is None:
        return None
    frags = list(Chem.GetMolFrags(mol, asMols=True))
    if not frags:
        return None
    frags.sort(key=lambda x: x.GetNumAtoms(), reverse=True)
    return frags[0]

def remove_map_nums(mol):
    """
    Remove atom map numbers from a molecule in place.
    """
    if mol is None:
        return None
    for atm in mol.GetAtoms():
        atm.SetAtomMapNum(0)
    return mol

def to_mol(obj):
    """
    Safely converts an object (SMILES string or Mol) to an RDKit Mol object.
    Returns a COPY of the molecule to avoid modifying the original.
    """
    if obj is None:
        return None
    if isinstance(obj, str):
        return Chem.MolFromSmiles(obj)
    elif isinstance(obj, Chem.Mol):
        return Chem.Mol(obj)  # Return a copy
    return None

def mol_to_base64(mol, size=(200, 200)):
    if mol is None:
        return ""
    try:
        img = Draw.MolToImage(mol, size=size)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except:
        return ""

def rxn_to_base64(rxn_smarts, size=(400, 150)):
    try:
        rxn = AllChem.ReactionFromSmarts(rxn_smarts, useSmiles=True)
        d2d = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        d2d.DrawReaction(rxn)
        d2d.FinishDrawing()
        return base64.b64encode(d2d.GetDrawingText()).decode("utf-8")
    except:
        return ""

# ------------------------------------------------------------------------------
# Core Logic
# ------------------------------------------------------------------------------

def process_mmp(df, pIC50_col='pIC50', smiles_col='SMILES', name_col='Name'):
    """
    Performs the MMP fragmentation and pairing analysis.
    """
    # 1. Clean and Standardize Molecules
    mols = []
    valid_indices = []
    
    st.write("Preprocessing molecules...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        smi = row[smiles_col]
        if pd.isna(smi): 
            continue
            
        mol = Chem.MolFromSmiles(str(smi))
        if mol:
            # Keep largest fragment (remove salts)
            mol = get_largest_fragment(mol)
            if mol:
                mols.append(mol)
                valid_indices.append(idx)
    
    if not valid_indices:
        return pd.DataFrame()

    df_clean = df.iloc[valid_indices].copy()
    df_clean['mol'] = mols
    
    # 2. Fragment Molecules
    st.write("Generating fragments...")
    row_list = []
    
    for i, row in tqdm(df_clean.iterrows(), total=len(df_clean)):
        mol = row['mol']
        name = row[name_col]
        pic50 = row[pIC50_col]
        smiles = row[smiles_col]
        
        # MMPA Fragmentation
        # We try-except the fragmentation to be safe against specific mol errors
        try:
            # Note: explicit resultsAsMols=True to get objects, but we handle strings below just in case
            frags = rdMMPA.FragmentMol(mol, maxCuts=1, resultsAsMols=True)
        except Exception:
            continue
            
        for core_raw, chain_raw in frags:
            # SAFETY CHECK: Convert inputs to Mols (handles Strings, Mols, or None)
            p1 = to_mol(core_raw)
            p2 = to_mol(chain_raw)
            
            # If either failed to convert, skip this fragment pair
            if p1 is None or p2 is None:
                continue
            
            # Remove map numbers (clean up dummies)
            p1 = remove_map_nums(p1)
            p2 = remove_map_nums(p2)
            
            # Sort by number of atoms to identify Core vs R-group
            if p1.GetNumAtoms() >= p2.GetNumAtoms():
                core_smi = Chem.MolToSmiles(p1)
                r_smi = Chem.MolToSmiles(p2)
            else:
                core_smi = Chem.MolToSmiles(p2)
                r_smi = Chem.MolToSmiles(p1)
                
            row_list.append([smiles, core_smi, r_smi, name, pic50])

    if not row_list:
        return pd.DataFrame()

    row_df = pd.DataFrame(row_list, columns=["SMILES", "Core", "R_group", "Name", "pIC50"])
    
    # 3. Find Pairs (Same Core, Different R-group)
    st.write("Identifying matched pairs...")
    delta_list = []
    
    # Group by Core
    groups = row_df.groupby("Core")
    
    for core, group in tqdm(groups):
        if len(group) >= 2:
            # Generate all pairs in this group
            for idx_a, idx_b in combinations(group.index, 2):
                row_a = group.loc[idx_a]
                row_b = group.loc[idx_b]
                
                # Skip identical compounds (by SMILES or Name)
                if row_a.SMILES == row_b.SMILES:
                    continue
                
                # Sort pair to ensure consistent direction A->B
                # Sorting by SMILES ensures A is always the same compound in the pair A-B vs B-A
                if row_a.SMILES > row_b.SMILES:
                    row_a, row_b = row_b, row_a
                
                delta = row_b.pIC50 - row_a.pIC50
                
                # Define Transform: R_group_A >> R_group_B
                transform = f"{row_a.R_group}>>{row_b.R_group}"
                
                delta_list.append({
                    "SMILES_1": row_a.SMILES, "Name_1": row_a.Name, "pIC50_1": row_a.pIC50, "R_group_1": row_a.R_group,
                    "SMILES_2": row_b.SMILES, "Name_2": row_b.Name, "pIC50_2": row_b.pIC50, "R_group_2": row_b.R_group,
                    "Core": core,
                    "Transform": transform,
                    "Delta": delta
                })
                
    delta_df = pd.DataFrame(delta_list)
    return delta_df
