from rdkit import Chem
import random
from rdkit.Chem import BRICS
import os, shutil
import numpy as np


def read_smiles_with_placeholders(file_path):
    valid_smiles = []
    with open(file_path, "r") as file:
        for line in file:
            smiles = line.strip()
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
                    valid_smiles.append(mol)
    return valid_smiles


def identify_placeholders(mol):
    output = []
    placeholder_indices = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            placeholder_indices.append(atom.GetIdx())
    for idx in placeholder_indices:
        atom = mol.GetAtomWithIdx(idx)
        smarts = atom.GetSmarts()
        number = smarts.strip("[]*")
        output.append([idx, int(number)])
    return output


def remove_placeholders(mol):
    placeholder = identify_placeholders(mol)
    placeholder_idx = [i[0] for i in placeholder]
    for idx in sorted(placeholder_idx, reverse=True):
        mol = delete_atom(mol, idx)
    return mol


def replace_placeholders(mol):
    rw_mol = Chem.RWMol(mol)
    for atom in rw_mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomicNum(6)
    mol = rw_mol.GetMol()
    Chem.SanitizeMol(mol)
    return mol


def delete_atom(mol, atom_idx):
    editable_mol = Chem.EditableMol(mol)
    editable_mol.RemoveAtom(atom_idx)
    new_mol = editable_mol.GetMol()
    Chem.SanitizeMol(new_mol)
    return new_mol


def placeholder_compatibility(placeholder1, placeholder2):
    if placeholder1 > placeholder2:
        placeholder_temp = placeholder1
        placeholder1 = placeholder2
        placeholder2 = placeholder_temp

    if placeholder1 == 1:
        if placeholder2 == 3 or placeholder2 == 5 or placeholder2 == 10:
            return True, Chem.BondType.SINGLE
    elif placeholder1 == 3:
        if placeholder2 == 1 or placeholder2 == 4 or placeholder2 == 13 or placeholder2 == 14 or placeholder2 == 15 or \
                placeholder2 == 16:
            return True, Chem.BondType.SINGLE
    elif placeholder1 == 4:
        if placeholder2 == 5 or placeholder2 == 11:
            return True, Chem.BondType.SINGLE
    elif placeholder1 == 5:
        if placeholder2 == 12 or placeholder2 == 13 or placeholder2 == 14 or \
                placeholder2 == 15 or placeholder2 == 16:
            return True, Chem.BondType.SINGLE
    elif placeholder1 == 6:
        if placeholder2 == 13 or placeholder2 == 14 or placeholder2 == 15 or placeholder2 == 16:
            return True, Chem.BondType.SINGLE
    elif placeholder1 == 7:
        if placeholder2 == 7:
            return True, Chem.BondType.DOUBLE
    elif placeholder1 == 8:
        if placeholder2 == 9 or placeholder2 == 10 or placeholder2 == 13 or placeholder2 == 14 or placeholder2 == 15 or \
                placeholder2 == 16:
            return True, Chem.BondType.SINGLE
    elif placeholder1 == 9:
        if placeholder2 == 13 or placeholder2 == 14 or placeholder2 == 15 or \
                placeholder2 == 16:
            return True, Chem.BondType.SINGLE
    elif placeholder1 == 10:
        if placeholder2 == 13 or placeholder2 == 14 or placeholder2 == 15 or \
                placeholder2 == 16:
            return True, Chem.BondType.SINGLE
    elif placeholder1 == 11:
        if placeholder2 == 13 or placeholder2 == 14 or placeholder2 == 15 or \
                placeholder2 == 16:
            return True, Chem.BondType.SINGLE
    elif placeholder1 == 13:
        if placeholder2 == 14 or placeholder2 == 15 or \
                placeholder2 == 16:
            return True, Chem.BondType.SINGLE
    elif placeholder1 == 14:
        if placeholder2 == 14 or placeholder2 == 15 or \
                placeholder2 == 16:
            return True, Chem.BondType.SINGLE
    elif placeholder1 == 15:
        if placeholder2 == 16:
            return True, Chem.BondType.SINGLE
    elif placeholder1 == 16:
        if placeholder2 == 16:
            return True, Chem.BondType.SINGLE

    return False, None


def combine_frags(mol, placeholder_idx, new_frag, new_frag_placeholder_idx):
    side_chain_atom_idx1 = mol.GetNumAtoms()
    combined_mol = Chem.RWMol(Chem.CombineMols(mol, new_frag))
    placeholder = combined_mol.GetAtomWithIdx(placeholder_idx)
    neighbours1 = [nbr.GetIdx() for nbr in placeholder.GetNeighbors()]
    new_frag_placeholder = combined_mol.GetAtomWithIdx(new_frag_placeholder_idx + side_chain_atom_idx1)
    neighbours = [nbr.GetIdx() for nbr in new_frag_placeholder.GetNeighbors()]
    combined_mol.AddBond(neighbours1[0], neighbours[0], Chem.rdchem.BondType.SINGLE)
    if placeholder_idx < neighbours1[0]:
        neighbours1[0] -= 1
    if placeholder_idx < neighbours[0]:
        neighbours[0] -= 1
    if new_frag_placeholder_idx + side_chain_atom_idx1 - 1 < neighbours1[0]:
        neighbours1[0] -= 1
    if new_frag_placeholder_idx + side_chain_atom_idx1 - 1 < neighbours[0]:
        neighbours[0] -= 1
    combined_mol.RemoveAtom(placeholder_idx)
    combined_mol.RemoveAtom(new_frag_placeholder_idx + side_chain_atom_idx1 - 1)
    prop_atom_idx = [neighbours1[0], neighbours[0]]
    final_mol = combined_mol.GetMol()
    Chem.SanitizeMol(final_mol)
    return final_mol, prop_atom_idx


def combine_frags_double_bond(mol, placeholder_idx, new_frag, new_frag_placeholder_idx):
    side_chain_atom_idx1 = mol.GetNumAtoms()
    combined_mol = Chem.RWMol(Chem.CombineMols(mol, new_frag))
    placeholder = combined_mol.GetAtomWithIdx(placeholder_idx)
    neighbours1 = [nbr.GetIdx() for nbr in placeholder.GetNeighbors()]
    new_frag_placeholder = combined_mol.GetAtomWithIdx(new_frag_placeholder_idx + side_chain_atom_idx1)
    neighbours = [nbr.GetIdx() for nbr in new_frag_placeholder.GetNeighbors()]
    combined_mol.AddBond(neighbours1[0], neighbours[0], Chem.rdchem.BondType.DOUBLE)
    if placeholder_idx < neighbours1[0]:
        neighbours1[0] -= 1
    if placeholder_idx < neighbours[0]:
        neighbours[0] -= 1
    if new_frag_placeholder_idx + side_chain_atom_idx1 - 1 < neighbours1[0]:
        neighbours1[0] -= 1
    if new_frag_placeholder_idx + side_chain_atom_idx1 - 1 < neighbours[0]:
        neighbours[0] -= 1
    combined_mol.RemoveAtom(placeholder_idx)
    combined_mol.RemoveAtom(new_frag_placeholder_idx + side_chain_atom_idx1 - 1)
    prop_atom_idx = [neighbours1[0], neighbours[0]]
    final_mol = combined_mol.GetMol()
    Chem.SanitizeMol(final_mol)
    return final_mol, prop_atom_idx


def remove_frags(mol, bond_number, placeholder_dict):
    mod_mol = Chem.RWMol(mol)
    bond_atom_idx = []
    for i in mol.GetAtoms():
        if i.HasProp(f'Bond{bond_number}'):
            bond_atom_idx.append(i.GetIdx())
    random.seed()
    a = random.random()
    if a > 0.5:
        placeholder_type = placeholder_dict[f'Bond{bond_number}']['placeholder_type1']
        placeholder_type = f'[{placeholder_type}*]'
        mod_mol.RemoveBond(bond_atom_idx[0], bond_atom_idx[1])
        placeholder = Chem.MolFromSmiles(placeholder_type)
        combined_mol = Chem.RWMol(Chem.CombineMols(mod_mol, placeholder))
        placeholder_idx = mol.GetNumAtoms()
        combined_mol.AddBond(bond_atom_idx[0], placeholder_idx, Chem.rdchem.BondType.SINGLE)
        fragments = Chem.GetMolFrags(combined_mol)[1]
        for i in sorted(fragments, reverse=True):
            combined_mol.RemoveAtom(i)
    else:
        placeholder_type = placeholder_dict[f'Bond{bond_number}']['placeholder_type2']
        placeholder_type = f'[{placeholder_type}*]'
        mod_mol.RemoveBond(bond_atom_idx[0], bond_atom_idx[1])
        placeholder = Chem.MolFromSmiles(placeholder_type)
        combined_mol = Chem.RWMol(Chem.CombineMols(mod_mol, placeholder))
        placeholder_idx = mol.GetNumAtoms()
        combined_mol.AddBond(bond_atom_idx[1], placeholder_idx, Chem.rdchem.BondType.SINGLE)
        fragments = Chem.GetMolFrags(combined_mol)[0]
        for i in sorted(fragments, reverse=True):
            combined_mol.RemoveAtom(i)
    final_mol = combined_mol.GetMol()
    Chem.SanitizeMol(final_mol)
    return final_mol


def create_frags(mol):
    fragments = BRICS.BRICSDecompose(mol)
    frag_list = []
    for f in fragments:
        frag_list.append(f)
    return frag_list


def create_frag_lib(mol_list):
    frag_list = []
    for mol in mol_list:
        frags = create_frags(mol)
        for f in frags:
            frag_list.append(f)
    canonical_smiles = {Chem.MolToSmiles(Chem.MolFromSmiles(smi)) for smi in frag_list}
    unique_smiles = list(set(canonical_smiles))
    for i in range(len(unique_smiles)):
        unique_smiles[i] = Chem.MolFromSmiles(unique_smiles[i])
    return unique_smiles


def save_structurs(name, step):
    filename = os.path.basename(name)
    best_structure = 0
    best_score = 0
    for i in range(5):
        path = f'/tmp/outputs/scores.model_idx_{i}.npz'
        data = np.load(path)
        score = data['aggregate_score'][0]
        if score > best_score:
            best_structure = i
            best_score = score
    path_to_cif = f'/tmp/outputs/pred.model_idx_{best_structure}.cif'
    currrent_dir = os.path.dirname(name)
    new_folder_name = f'{filename}'
    destination_dir = os.path.join(currrent_dir, new_folder_name)
    if not os.path.isdir(destination_dir):
        os.makedirs(destination_dir)
    destination_file = os.path.join(destination_dir, os.path.basename(path_to_cif))
    shutil.copy(path_to_cif, destination_file)
    new_file_name = f'{filename}_{step}.cif'
    renamed_file_path = os.path.join(destination_dir, new_file_name)
    print(renamed_file_path)
    os.rename(destination_file, renamed_file_path)
