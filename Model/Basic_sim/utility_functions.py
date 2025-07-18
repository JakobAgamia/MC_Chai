from rdkit import Chem
from pathlib import Path
import numpy as np
import torch
from chai_lab.chai1 import run_inference
import random
from rdkit.Chem import AllChem, DataStructs
import time
from rdkit.Chem.rdShapeHelpers import ShapeTanimotoDist
from itertools import product
from rdkit.Contrib.SA_Score import sascorer
from rdkit.Chem import QED
from rdkit.Chem import Descriptors, Crippen, Lipinski
import shutil, os


def add_side_chains(molecule, side_chain, bonding_position):
    random.seed()
    op = random.random()
    try:
        if op > 0.3:
            new = add_side_chain_single_bond(molecule, side_chain, bonding_position)
            return new
        else:
            new = add_side_chains_double_bond(molecule, side_chain, bonding_position)
            return new
    except Exception as e:
        atom = molecule.GetAtomWithIdx(bonding_position)
        if atom.IsInRing():
            new = add_side_chains_bond_adjustment_ring(molecule, side_chain, bonding_position)
            return new


def add_side_chain_single_bond(molecule, side_chain, bonding_position):
    combined_mol = Chem.RWMol(Chem.CombineMols(molecule, side_chain))
    side_chain_atom_idx = molecule.GetNumAtoms()
    combined_mol.AddBond(bonding_position, side_chain_atom_idx, Chem.rdchem.BondType.SINGLE)
    final_mol = combined_mol.GetMol()
    Chem.SanitizeMol(final_mol)
    return final_mol


def add_side_chains_double_bond(molecule, side_chain, bonding_position):
    combined_mol = Chem.RWMol(Chem.CombineMols(molecule, side_chain))
    side_chain_atom_idx = molecule.GetNumAtoms()
    combined_mol.AddBond(bonding_position, side_chain_atom_idx, Chem.rdchem.BondType.DOUBLE)
    final_mol = combined_mol.GetMol()
    Chem.SanitizeMol(final_mol)
    return final_mol


def add_side_chains_bond_adjustment_ring(molecule, side_chain, bonding_position):
    combined_mol = Chem.RWMol(Chem.CombineMols(molecule, side_chain))
    side_chain_atom_idx = molecule.GetNumAtoms()
    combined_mol.AddBond(bonding_position, side_chain_atom_idx, Chem.rdchem.BondType.SINGLE)
    final_mol = combined_mol.GetMol()
    bonds = get_external_bonds_ring(molecule, bonding_position)
    combinations = generate_all_combinations(final_mol, bonds)
    combinations_list = []
    for i in combinations:
        combinations_list.append(i)
    random.seed()
    random.shuffle(combinations_list)
    for mol in combinations_list:
        try:
            Chem.SanitizeMol(mol)
            final_mol = mol
            break
        except Exception as e:
            continue
    Chem.SanitizeMol(final_mol)
    return final_mol


def get_external_bonds_ring(molecule, atom_idx):
    ring_info = molecule.GetRingInfo()
    all_rings = ring_info.AtomRings()
    ring_of_interest = None
    for ring in all_rings:
        if atom_idx in ring:
            ring_of_interest = ring
            break
    ring_atom_set = set(ring_of_interest)
    external_bonds = []
    for atom_idx in ring_of_interest:
        atom = molecule.GetAtomWithIdx(atom_idx)
        for bond in atom.GetBonds():
            other_atom = bond.GetOtherAtom(atom)
            other_idx = other_atom.GetIdx()
            if other_idx not in ring_atom_set:
                external_bonds.append(bond.GetIdx())
    return external_bonds


def generate_all_combinations(mol, bond_idx):
    single_bond_idxs = bond_idx
    from itertools import chain, combinations
    def all_subsets(iterable):
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))
    for subset in all_subsets(single_bond_idxs):
        new_mol = Chem.RWMol(mol)
        for bond_idx in subset:
            bond = new_mol.GetBondWithIdx(bond_idx)
            bond.SetBondType(Chem.BondType.DOUBLE)
        yield new_mol.GetMol()


def modify_implicit_hydrogens(molecule, atom_idx, num_hydrogens):
    editable_mol = Chem.RWMol(molecule)
    atom = editable_mol.GetAtomWithIdx(atom_idx)
    atom.SetNumExplicitHs(num_hydrogens)
    return editable_mol.GetMol()


def change_atom_type(mol, atom_idx, atom_number):
    atom = mol.GetAtomWithIdx(atom_idx)
    atom.SetAtomicNum(atom_number)
    return mol


def create_combinations(mol, atoms_in_ring):
    s = False
    combinations = generate_combinations(mol, atoms_in_ring)
    random.seed()
    random.shuffle(combinations)
    for combination in combinations:
        new_mol = mol
        try:
            for change in combination:
                if change[1] == 'bond':
                    new_mol = change_bond_type(new_mol, change[0], change[2])
                else:
                    new_mol = modify_implicit_hydrogens(new_mol, change[0], change[2])
            Chem.SanitizeMol(new_mol)
            s = True
            break
        except Exception as e:
            continue
    if s:
        return new_mol
    else:
        raise TypeError


def change_bond_type(mol, bond_idx, new_bond_type):
    rw_mol = Chem.RWMol(mol)
    bond_index = bond_idx
    bond = rw_mol.GetBondWithIdx(bond_index)
    bond.SetBondType(new_bond_type)
    bond.SetIsAromatic(True)
    modified_mol = rw_mol.GetMol()
    return modified_mol


def generate_combinations(mol,  atoms_in_ring):
    options = []
    for atom_idx in atoms_in_ring:
        external_bonds = get_external_bonds_atom(mol, atom_idx)
        if external_bonds:
            for i in external_bonds:
                options.append(bond_variations(i))
        else:
            options.append(hydrogen_variations(atom_idx))
    all_combinations = list(product(*options))
    return all_combinations


def get_external_bonds_atom(molecule, atom_idx):
    external_bonds = []
    atom = molecule.GetAtomWithIdx(atom_idx)
    for bond in atom.GetBonds():
        other_atom = bond.GetOtherAtom(atom)
        if not other_atom.IsInRing():
            external_bonds.append(bond.GetIdx())
    return external_bonds


def bond_variations(bond):
    return [(bond, 'bond', Chem.BondType.SINGLE), (bond, 'bond', Chem.BondType.DOUBLE)]


def hydrogen_variations(atom):
    return [(atom, 'hydrogen', 0), (atom, 'hydrogen', 1), (atom, 'hydrogen', 2)]


def get_chain_index_calculation(atom_idx, mol, prev_idx=None):
    idx_list = [atom_idx]
    previous_idx = [prev_idx]
    atom = mol.GetAtomWithIdx(atom_idx)
    bonded_atom_indices = [bond.GetOtherAtomIdx(atom_idx) for bond in atom.GetBonds()]
    bonded_atom_indices = [a for a in bonded_atom_indices if a != previous_idx[0]]
    previous_idx.append(atom_idx)
    if len(bonded_atom_indices) == 1:
        x = []
        while not atom.IsInRing():
            atom = mol.GetAtomWithIdx(bonded_atom_indices[-1])
            idx_list.append(bonded_atom_indices[-1])
            previous_idx.append(bonded_atom_indices[-1])
            bonded_atom_indices = [bond.GetOtherAtomIdx(bonded_atom_indices[-1]) for bond in atom.GetBonds()]
            if len(bonded_atom_indices) == 1:
                break
            bonded_atom_indices = [a for a in bonded_atom_indices if a != previous_idx[-2]]
            if len(previous_idx) >= 2:
                bonded_atom_indices = [a for a in bonded_atom_indices if a != previous_idx[-3]]
            if atom.IsInRing():
                random.seed()
                rand = random.random()
                if rand <= 0.5:
                    idx_list.append(bonded_atom_indices[-1])
                    atom_in_ring = mol.GetAtomWithIdx(bonded_atom_indices[-1])
                    neighbours = [nbr.GetIdx() for nbr in atom_in_ring.GetNeighbors()]
                    for i in neighbours:
                        if not mol.GetAtomWithIdx(i).IsInRing():
                            idx_list.append(i)
                            second_neighbours = [nbr.GetIdx() for nbr in atom_in_ring.GetNeighbors()]
                            for i2 in second_neighbours:
                                if not mol.GetAtomWithIdx(i2).IsInRing():
                                    if i2 not in idx_list:
                                        idx_list.append(i2)
                    print(idx_list)
                else:
                    idx_list.append(bonded_atom_indices[0])
                    atom_in_ring = mol.GetAtomWithIdx(bonded_atom_indices[0])
                    neighbours = [nbr.GetIdx() for nbr in atom_in_ring.GetNeighbors()]
                    for i in neighbours:
                        if not mol.GetAtomWithIdx(i).IsInRing():
                            idx_list.append(i)
                            second_neighbours = [nbr.GetIdx() for nbr in atom_in_ring.GetNeighbors()]
                            for i2 in second_neighbours:
                                if not mol.GetAtomWithIdx(i2).IsInRing():
                                    if i2 not in idx_list:
                                        idx_list.append(i2)
                    print(idx_list)
            if not atom.IsInRing():
                if len(bonded_atom_indices) != 1:
                    for b in bonded_atom_indices:
                        result = get_chain_index_calculation(b, mol, previous_idx[-1])
                        x.append(result)
                    break
            else:
                break
        if x:
            return [idx_list] + x
        else:
            return idx_list
    elif len(bonded_atom_indices) >= 1:
        k = [atom_idx]
        if not atom.IsInRing():
            for b in bonded_atom_indices:
                result = get_chain_index_calculation(b, mol, previous_idx[-1])
                k.append(result)
        return k
    else:
        return idx_list


def get_chain_index(atom_idx, mol):
    results = get_chain_index_calculation(atom_idx, mol)
    if isinstance(results[0], list):
        results = combine_lists_recursively(results)
    else:
        results = [results]
    return results


def combine_lists_recursively(lst):
    prev_element = lst[0]
    if not isinstance(prev_element, list):
        prev_element = [prev_element]
    final_list = []
    for i in lst[1:]:
        if isinstance(i, list):
            if any(isinstance(l, list) for l in i):
                s = combine_lists_recursively(i)
                for k in s:
                    if not isinstance(k, list):
                        k = [k]
                    final_list.append(prev_element + k)
            else:
                if not isinstance(i, list):
                        i = [i]
                final_list.append(prev_element + i)
        else:
                if not isinstance(i, list):
                        i = [i]
                final_list.append(prev_element + i)
    return final_list


def find_ring_pairs(mol):
    starting_points = identify_edge_atoms(mol)
    index_list = []
    for i in starting_points:
        chains = get_chain_index(i, mol)
        for chain in chains:
            if len(chain) >= 5:
                for c in range(len(chain) - 4):
                    index_list.append([chain[c], chain[c+4]])
            if len(chain) >= 6:
                for c in range(len(chain) - 5):
                    index_list.append([chain[c], chain[c+5]])
    return index_list


def identify_edge_atoms(mol):
    removable_atom_list = []
    for atom in mol.GetAtoms():
        if atom.GetDegree() == 1:
            atom_symbol = atom.GetSymbol()
            atom_index = atom.GetIdx()
            removable_atom_list.append(atom_index)
    return removable_atom_list


def can_create_ring(mol):
    l = find_ring_pairs(mol)
    new_lig = []
    for i in l:
        try:
            new_lig = create_ring(i[0], i[1], mol)
            break
        except Exception as e:
            continue
    if not new_lig:
        return False
    else:
        return True


def create_ring(idx1, idx2, mol):
    atom_idx_mol1 = idx1
    atom_idx_mol2 = idx2
    combined_rw_mol = Chem.RWMol(mol)
    combined_rw_mol.AddBond(atom_idx_mol1, atom_idx_mol2, Chem.BondType.SINGLE)
    final_molecule = combined_rw_mol.GetMol()
    Chem.SanitizeMol(final_molecule)
    return final_molecule


def bonds_in_ring(mol):
    bonds = []
    for i in range(mol.GetNumBonds()):
        if mol.GetBondWithIdx(i).IsInRing():
            bonds.append(i)
    return bonds


def remove_bond(mol, bond_idx):
    rw_mol = Chem.RWMol(mol)
    bond_index = bond_idx
    bond = rw_mol.GetBondWithIdx(bond_index)
    atom1_idx = bond.GetBeginAtomIdx()
    atom2_idx = bond.GetEndAtomIdx()
    rw_mol.RemoveBond(atom1_idx, atom2_idx)
    for a in rw_mol.GetAtoms():
        if (not a.IsInRing()) and a.GetIsAromatic():
            a.SetIsAromatic(False)
    for a in rw_mol.GetBonds():
        if (not a.IsInRing()) and a.GetIsAromatic():
            a.SetIsAromatic(False)
    mol = rw_mol.GetMol()
    Chem.SanitizeMol(mol)
    return mol


def can_remove_ring(mol):
    bonds = bonds_in_ring(mol)
    s = False
    for i in bonds:
        try:
            new_mol = remove_bond(mol, i)
            Chem.SanitizeMol(new_mol)
            s = True
            break
        except Exception as e:
                continue
    if s:
        return True
    else:
        return False


def is_ring_aromatic_bonds(mol, bond_ring):
    return all(mol.GetBondWithIdx(bond_idx).GetIsAromatic() for bond_idx in bond_ring)


def turn_ring_aromatic(mol, bond_idx_list):
    for i in bond_idx_list:
        mol = change_bond_type(mol, i, Chem.BondType.AROMATIC)
        rw_mol = Chem.RWMol(mol)
        bond = rw_mol.GetBondWithIdx(i)
        atom_id = bond.GetBeginAtomIdx()
        atom = rw_mol.GetAtomWithIdx(atom_id)
        atom.SetIsAromatic(True)
        modified_mol = rw_mol.GetMol()
    Chem.SanitizeMol(modified_mol)
    return modified_mol


def turn_ring_non_aromatic(mol, ring):
    mol_edit = Chem.RWMol(mol)
    for bond_idx in ring:
        bond = mol_edit.GetBondWithIdx(bond_idx)
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    for bond_idx in ring:
        bond = mol_edit.GetBondWithIdx(bond_idx)
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        atom1.SetIsAromatic(False)
        atom2.SetIsAromatic(False)
    Chem.SanitizeMol(mol_edit)
    return mol_edit


def add_atom_in_chain(mol, atom_idx, new_addition):
    combined_mol = Chem.RWMol(Chem.CombineMols(mol, new_addition))
    atom = combined_mol.GetAtomWithIdx(atom_idx)
    neighbours = [nbr.GetIdx() for nbr in atom.GetNeighbors()]
    second_atom_idx = random.choice(neighbours)
    combined_mol.RemoveBond(atom_idx, second_atom_idx)
    num_atoms = mol.GetNumAtoms()
    combined_mol.AddBond(atom_idx, num_atoms, Chem.rdchem.BondType.SINGLE)
    combined_mol.AddBond(second_atom_idx, num_atoms, Chem.rdchem.BondType.SINGLE)
    final_mol = combined_mol.GetMol()
    Chem.SanitizeMol(final_mol)
    return final_mol


def are_atoms_not_in_same_ring(mol, atom_idx1, atom_idx2):
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if atom_idx1 in ring and atom_idx2 in ring:
            return False
    return True


def move_junction(mol, atom_idx1, atom_idx2):
    atom_center = mol.GetAtomWithIdx(atom_idx1)
    neighbours = [nbr.GetIdx() for nbr in atom_center.GetNeighbors()]
    neighbours = [x for x in neighbours if x != atom_idx2]
    second_neighbours = []
    for i in neighbours:
        atom = mol.GetAtomWithIdx(i)
        s = [nbr.GetIdx() for nbr in atom.GetNeighbors()]
        second_neighbours = list(set(second_neighbours + s))
    second_neighbours = [x for x in second_neighbours if x != atom_idx1]
    third_neighbours = []
    for i in second_neighbours:
        atom = mol.GetAtomWithIdx(i)
        s = [nbr.GetIdx() for nbr in atom.GetNeighbors()]
        third_neighbours = list(set(third_neighbours + s))
    third_neighbours = [x for x in third_neighbours if x not in neighbours]
    fourth_neighbours = []
    for i in third_neighbours:
        atom = mol.GetAtomWithIdx(i)
        s = [nbr.GetIdx() for nbr in atom.GetNeighbors()]
        fourth_neighbours = list(set(fourth_neighbours + s))
    fourth_neighbours = [x for x in fourth_neighbours if x not in second_neighbours]
    new_positions = list(set(neighbours + second_neighbours + third_neighbours + fourth_neighbours))
    bond = mol.GetBondBetweenAtoms(atom_idx1, atom_idx2)
    bond_type = bond.GetBondType()
    random.seed()
    new_position = random.choice(new_positions)
    combined_mol = Chem.EditableMol(mol)
    combined_mol.RemoveBond(atom_idx1, atom_idx2)
    combined_mol.AddBond(atom_idx2, new_position, bond_type)
    final_mol = combined_mol.GetMol()
    return final_mol


def remove_atom(mol, atom_idx):
    editable_mol = Chem.EditableMol(mol)
    editable_mol.RemoveAtom(atom_idx)
    new_mol = editable_mol.GetMol()
    return new_mol


def remove_atom_in_chain(mol, atom_idx):
    atom = mol.GetAtomWithIdx(atom_idx)
    neighbours = [nbr.GetIdx() for nbr in atom.GetNeighbors()]
    combined_mol = Chem.EditableMol(mol)
    for i in neighbours:
        combined_mol.RemoveBond(atom_idx, i)
    combined_mol.AddBond(neighbours[0], neighbours[1], Chem.rdchem.BondType.SINGLE)
    combined_mol.RemoveAtom(atom_idx)
    final_mol = combined_mol.GetMol()
    return final_mol


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
