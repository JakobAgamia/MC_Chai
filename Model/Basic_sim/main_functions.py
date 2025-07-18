from rdkit import Chem
import random
import numpy as np
from score_functions import get_biased_ligand_score
from utility_functions import add_side_chains
from utility_functions import modify_implicit_hydrogens
from utility_functions import change_atom_type
from utility_functions import create_combinations
from utility_functions import can_remove_ring
from utility_functions import bonds_in_ring
from utility_functions import remove_bond
from utility_functions import change_bond_type
from utility_functions import can_create_ring
from utility_functions import find_ring_pairs
from utility_functions import create_ring
from utility_functions import is_ring_aromatic_bonds
from utility_functions import turn_ring_aromatic
from utility_functions import turn_ring_non_aromatic
from utility_functions import add_atom_in_chain
from utility_functions import are_atoms_not_in_same_ring
from utility_functions import move_junction
from utility_functions import remove_atom
from utility_functions import remove_atom_in_chain


def step(protein, lig, previous_score, max_len, params, prob, log_name, params_score, restraints, s, lig_score_list):
    output = []
    random.seed()
    num_op_list = [1, 2, 3]
    num_op_probabilities = [0.8, 0.0, 0.0]
    num_ops = random.choices(num_op_list, num_op_probabilities)
    ligand = lig
    for i in range(num_ops[0]):
        random.seed()
        operation = random.random()
        len = lig.GetNumAtoms()
        if operation >= params[0]:
            new = step_add_c(lig, max_len)
            ligand = new
            output.append('add')
        elif params[0] >= operation >= params[1]:
            new = step_change_atom_type(lig)
            ligand = new
            output.append('change type')
        elif params[1] >= operation >= params[2]:
            new = step_remove_ring(lig)
            ligand = new
            output.append('remove ring')
        elif params[2] >= operation >= params[3]:
            new = step_change_bond_type(lig)
            ligand = new
            output.append('change bond type')
        elif params[3] >= operation >= params[4]:
            new = step_form_new_ring(lig)
            ligand = new
            output.append('form new ring')
        elif params[4] >= operation >= params[5]:
            new = step_turn_ring_aromatic(lig)
            ligand = new
            output.append('turn ring aromatic')
        elif params[5] >= operation >= params[6]:
            new = step_switch(lig, max_len)
            ligand = new
            output.append('switch')
        elif params[6] >= operation >= params[7]:
            new = step_move_junction(lig)
            ligand = new
            output.append('move_junction')
        else:
            new = step_remove_atom(lig)
            ligand = new
            output.append('remove')
    if Chem.MolToSmiles(lig) != Chem.MolToSmiles(ligand):
        try:
            with open(log_name, 'a') as test_sim:
                test_sim.write(f'Step{s}: {output}, Previous ligand: {Chem.MolToSmiles(lig)}, Changed ligand: {Chem.MolToSmiles(ligand)}')
            lig_known = False
            for lig_score in lig_score_list:
                if Chem.MolToSmiles(ligand) == lig_score[0]:
                    score = lig_score[1]
                    lig_known = True
            if not lig_known:
                score = get_biased_ligand_score(Chem.MolToSmiles(ligand), protein, '/tmp/outputs', params_score, restraints)
                lig_score_list.append([Chem.MolToSmiles(ligand), score])
            if score > previous_score:
                lig = ligand
                previous_score = score
            else:
                probability = np.exp((score-previous_score)*prob)
                random.seed()
                rand = random.random()
                if rand < probability:
                    lig = ligand
                    previous_score = score
            with open(log_name, 'a') as test_sim:
                test_sim.write(f', New ligand: {Chem.MolToSmiles(lig)} Score: {score}\n')
            return lig, previous_score
        except Exception:
            return lig, previous_score
    else:
        return step(protein, lig, previous_score, max_len, params, prob, log_name, params_score, restraints, s, lig_score_list)


def step_add_c(lig, max_len):
    side_chain1 = Chem.MolFromSmiles('C')
    side_chain2 = Chem.MolFromSmiles('N')
    side_chain3 = Chem.MolFromSmiles('O')
    side_chain4 = Chem.MolFromSmiles('S')
    random.seed()
    side_chains = [side_chain1, side_chain2, side_chain3, side_chain4]
    side_chain_probabilities = [1, 0.25, 0.25, 0.03]
    side_chain = random.choices(side_chains, side_chain_probabilities)[0]
    length = lig.GetNumAtoms()
    side_chain_length = side_chain.GetNumAtoms()
    total_length = side_chain_length + length
    if max_len > total_length:
        binding_positions = [i for i in range(length)]
        s = False
        random.seed()
        random.shuffle(binding_positions)
        for i in binding_positions:
            binding_atom = lig.GetAtomWithIdx(i)
            rand = random.random()
            if side_chain == side_chain1 or binding_atom.GetAtomicNum() == 6 or rand > 0.9:
                try:
                    new_ligand = add_side_chains(lig, side_chain, i)
                    Chem.SanitizeMol(new_ligand)
                    s = True
                    break
                except Exception as e:
                    try:
                        lig2 = modify_implicit_hydrogens(lig, i, 0)
                        new_ligand = add_side_chains(lig2, side_chain, i)
                        Chem.SanitizeMol(new_ligand)
                        s = True
                        break
                    except Exception as e:
                        continue
            else:
                continue
        if s:
            lig = new_ligand
            return lig
        else:
            return lig
    else:
        return lig


def step_change_atom_type(lig):
    rw_mol = Chem.RWMol(lig)
    atoms_list = []
    for a in rw_mol.GetAtoms():
        atoms_list.append(a.GetIdx())
    random.seed()
    atom_types = [6, 7, 8, 9, 16, 15]
    atom_probabilities = [1, 0.25, 0.25, 0.03, 0.03, 0.01]
    atom_choice = random.choices(atom_types, atom_probabilities)
    s = False
    lig_copy = Chem.Mol(lig)
    if atoms_list:
        random.seed()
        random.shuffle(atoms_list)
        for atom_idx in atoms_list:
            lig = Chem.Mol(lig_copy)
            atom = lig.GetAtomWithIdx(atom_idx)
            neighbours = [nbr.GetIdx() for nbr in atom.GetNeighbors()]
            neighbours_c = True
            for neighbour in neighbours:
                neighbour_atom = lig.GetAtomWithIdx(neighbour)
                if neighbour_atom.GetAtomicNum() != 6:
                    neighbours_c = False
            rand = random.random()
            print(rand)
            if atom_choice[0] == 6 or rand > 0.9 or neighbours_c:
                if not atom.IsInRing():
                    try:
                        new_ligand = change_atom_type(lig, atom_idx, atom_choice[0])
                        if Chem.MolToSmiles(new_ligand) != Chem.MolToSmiles(lig):
                            Chem.SanitizeMol(new_ligand)
                            test_lig = Chem.MolFromSmiles(Chem.MolToSmiles(new_ligand))
                            Chem.SanitizeMol(test_lig)
                            s = True
                            break
                        else:
                            continue
                    except Exception as e:
                        continue
                else:
                    ring_info = lig.GetRingInfo()
                    all_rings = ring_info.AtomRings()
                    ring_of_interest = None
                    for ring in all_rings:
                        if atom_idx in ring:
                            ring_of_interest = ring
                    try:
                        new_ligand = change_atom_type(lig, atom_idx, atom_choice[0])
                        Chem.SanitizeMol(new_ligand)
                        test_lig = Chem.MolFromSmiles(Chem.MolToSmiles(new_ligand))
                        Chem.SanitizeMol(test_lig)
                        s = True
                        break
                    except Exception as e:
                        try:
                            new_ligand = change_atom_type(lig, atom_idx, atom_choice[0])
                            new_ligand = create_combinations(new_ligand, ring_of_interest)
                            Chem.SanitizeMol(new_ligand)
                            test_lig = Chem.MolFromSmiles(Chem.MolToSmiles(new_ligand))
                            Chem.SanitizeMol(test_lig)
                            s = True
                            break
                        except Exception:
                            continue
            else:
                continue
        if s:
            lig = new_ligand
            return lig
        else:
            return lig
    else:
        return lig


def step_remove_ring(lig):
    if can_remove_ring(lig):
        l = bonds_in_ring(lig)
        random.seed()
        random.shuffle(l)
        s = False
        for bond in l:
            try:
                new_ligand = remove_bond(lig, bond)
                rw_mol = Chem.RWMol(new_ligand)
                for a in rw_mol.GetBonds():
                    if not a.IsInRing():
                        a.SetBondType(Chem.BondType.SINGLE)
                new_ligand = rw_mol.GetMol()
                Chem.SanitizeMol(new_ligand)
                s = True
                break
            except Exception as e:
                print(e)
                continue
        if s:
            lig = new_ligand
            return lig
        else:
            return lig
    else:
        return lig


def step_change_bond_type(lig):
    rw_mol = Chem.RWMol(lig)
    bonds_list = []
    for a in rw_mol.GetBonds():
        if not a.IsInRing():
            bonds_list.append(a.GetIdx())
    random.seed()
    bond_types = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE]
    bond_probabilities = [0.5, 0.5]
    bond_choice = random.choices(bond_types, bond_probabilities)
    s = False
    if bonds_list:
        random.seed()
        random.shuffle(bonds_list)
        for bond in bonds_list:
            try:
                new_ligand = change_bond_type(lig, bond, bond_choice[0])
                Chem.SanitizeMol(new_ligand)
                s = True
                break
            except Exception as e:
                continue
        if s:
            lig = new_ligand
            return lig
        else:
            return lig
    else:
        return lig


def step_form_new_ring(lig):
    if can_create_ring(lig):
        l = find_ring_pairs(lig)
        random.seed()
        random.shuffle(l)
        s = False
        for pair in l:
            try:
                new_ligand = create_ring(pair[0], pair[1], lig)
                s = True
                break
            except Exception as e:
                continue
        if s:
            lig = new_ligand
            return lig
        else:
            return lig
    else:
        return lig


def step_turn_ring_aromatic(lig):
    ring_info = lig.GetRingInfo()
    rings = ring_info.BondRings()
    rings_list = []
    for i in rings:
        rings_list.append(i)
    s = False
    if rings:
        random.seed()
        random.shuffle(rings_list)
        for ring in rings_list:
            try:
                if not is_ring_aromatic_bonds(lig, ring):
                    new_ligand = turn_ring_aromatic(lig, ring)
                    s = True
                    break
                else:
                    new_ligand = turn_ring_non_aromatic(lig, ring)
                    s = True
                    break
            except Exception as e:
                print(e)
                continue
        if s:
            lig = new_ligand
            return lig
        else:
            return lig
    else:
        return lig


def step_switch(lig, max_len):
    s = False
    r = False
    rw_mol = Chem.RWMol(lig)
    atoms_list = []
    for a in rw_mol.GetAtoms():
        atoms_list.append(a.GetIdx())
    random.seed()
    random.shuffle(atoms_list)
    side_chain1 = Chem.MolFromSmiles('C')
    side_chain2 = Chem.MolFromSmiles('N')
    side_chain3 = Chem.MolFromSmiles('O')
    side_chain4 = Chem.MolFromSmiles('S')
    random.seed()
    side_chains = [side_chain1, side_chain2, side_chain3, side_chain4]
    side_chain_probabilities = [0.1, 0.025, 0.025, 0.01]
    side_chain = random.choices(side_chains, side_chain_probabilities)[0]
    length = lig.GetNumAtoms()
    side_chain_length = side_chain.GetNumAtoms()
    total_length = side_chain_length + length
    if max_len > total_length:
        for atom_idx in atoms_list:
            atom = lig.GetAtomWithIdx(atom_idx)
            neighbours = [nbr.GetIdx() for nbr in atom.GetNeighbors()]
            second_atom_idx = random.choice(neighbours)
            second_atom = lig.GetAtomWithIdx(second_atom_idx)
            neighbours_c = False
            if atom.GetAtomicNum() == 6 and second_atom.GetAtomicNum() == 6:
                neighbours_c = True
            rand = random.random()
            if side_chain == side_chain1 or rand > 0.9 or neighbours_c:
                if not atom.IsInRing():
                    try:
                        new_ligand = add_atom_in_chain(lig, atom_idx, side_chain, second_atom_idx)
                        s = True
                        break
                    except Exception as e:
                        continue
                else:
                    ring_info = lig.GetRingInfo()
                    all_rings = ring_info.AtomRings()
                    ring_of_interest = None
                    for ring in all_rings:
                        if atom_idx in ring:
                            ring_of_interest = ring
                    total_lenght = len(ring_of_interest) + 1
                    if total_lenght < 7:
                        try:
                            new_ligand = add_atom_in_chain(lig, atom_idx, side_chain, second_atom_idx)
                            ring_of_interest = list(ring_of_interest)
                            ring_of_interest.append(length)
                            new_ligand = create_combinations(new_ligand, ring_of_interest)
                            Chem.SanitizeMol(new_ligand)
                            s = True
                            break
                        except Exception as e:
                            continue
                    else:
                        return lig
            else:
                continue
        if s:
            lig = new_ligand
            return lig
        else:
            return lig
    else:
        return lig


def step_move_junction(lig):
    length = lig.GetNumAtoms()
    binding_positions = [i for i in range(length)]
    s = False
    random.seed()
    random.shuffle(binding_positions)
    for atom_idx1 in binding_positions:
        atom_center = lig.GetAtomWithIdx(atom_idx1)
        neighbours = [nbr.GetIdx() for nbr in atom_center.GetNeighbors()]
        if len(neighbours) > 2:
            random.seed()
            atom_idx2 = random.choice(neighbours)
            if are_atoms_not_in_same_ring(lig, atom_idx1, atom_idx2):
                try:
                    new_ligand = move_junction(lig, atom_idx1, atom_idx2)
                    Chem.SanitizeMol(new_ligand)
                    s = True
                    break
                except Exception as e:
                    continue
            else:
                continue
        else:
            continue
    if s:
        lig = new_ligand
        return lig
    else:
        return lig


def step_remove_atom(lig):
    s = False
    r = False
    rw_mol = Chem.RWMol(lig)
    atoms_list = []
    for a in rw_mol.GetAtoms():
        atoms_list.append(a.GetIdx())
    random.seed()
    random.shuffle(atoms_list)
    l = []
    for atom_idx in atoms_list:
        atom = lig.GetAtomWithIdx(atom_idx)
        neighbours = [nbr.GetIdx() for nbr in atom.GetNeighbors()]
        if not atom.IsInRing():
            if len(neighbours) == 1:
                try:
                    new_ligand = remove_atom(lig, atom_idx)
                    Chem.SanitizeMol(new_ligand)
                    l.append(atom_idx)
                    l.append('no_ring_1')
                    s = True
                    break
                except Exception as e:
                    continue
            if len(neighbours) == 2:
                try:
                    new_ligand = remove_atom_in_chain(lig, atom_idx)
                    Chem.SanitizeMol(new_ligand)
                    l.append(atom_idx)
                    l.append('no_ring_2')
                    s = True
                    break
                except Exception as e:
                    continue
        else:
            ring_info = lig.GetRingInfo()
            all_rings = ring_info.AtomRings()
            ring_of_interest = None
            for ring in all_rings:
                if atom_idx in ring:
                    ring_of_interest = ring
            if len(ring_of_interest) >= 6:
                if len(neighbours) == 2:
                    try:
                        new_ligand = remove_atom_in_chain(lig, atom_idx)
                        Chem.SanitizeMol(new_ligand)
                        s = True
                        r = True
                        break
                    except Exception as e:
                        try:
                            new_ligand = remove_atom_in_chain(lig, atom_idx)
                            max_value = max(ring_of_interest)
                            ring_of_interest = list(ring_of_interest)
                            ring_of_interest.remove(max_value)
                            new_ligand = create_combinations(new_ligand, ring_of_interest)
                            Chem.SanitizeMol(new_ligand)
                            s = True
                            break
                        except Exception as e:
                            continue
    if s:
        lig = new_ligand
        return lig
    else:
        return lig
