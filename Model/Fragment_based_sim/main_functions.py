from rdkit import Chem
import random
from score_functions import get_biased_ligand_score
import numpy as np
from utility_functions import identify_placeholders
from utility_functions import placeholder_compatibility
from utility_functions import combine_frags
from utility_functions import combine_frags_double_bond
from utility_functions import remove_placeholders
from utility_functions import replace_placeholders
from utility_functions import remove_frags


def step_combine(lig, frag_lib, previous_score, protein, placeholder_dict, prob, params_score, restraints):
    placeholders = identify_placeholders(lig)
    random.seed()
    random.shuffle(placeholders)
    placeholder = placeholders[0]
    placeholder_idx = placeholder[0]
    placeholder_type = placeholder[1]
    random.seed()
    random.shuffle(frag_lib)
    s = False
    for i in frag_lib:
        frag_placeholders = identify_placeholders(i)
        random.seed()
        random.shuffle(frag_placeholders)
        for placeholder in frag_placeholders:
            frag_placeholder_idx = placeholder[0]
            frag_placeholder_type = placeholder[1]
            compatibility = placeholder_compatibility(placeholder_type, frag_placeholder_type)
            if compatibility[0]:
                if compatibility[1] == Chem.BondType.SINGLE:
                    try:
                        new = combine_frags(lig, placeholder_idx, i, frag_placeholder_idx)
                        new_ligand = new[0]
                        prop_idx = new[1]
                        s = True
                        break
                    except Exception as e:
                        continue
                else:
                    try:
                        new = combine_frags_double_bond(lig, placeholder_idx, i, frag_placeholder_idx)
                        new_ligand = new[0]
                        prop_idx = new[1]
                        s = True
                        break
                    except Exception as e:
                        continue
        if s:
            break
    if s:
        try:
            new_ligand_no_placeholders = remove_placeholders(new_ligand)
            new_score = get_biased_ligand_score(Chem.MolToSmiles(new_ligand_no_placeholders), protein, '/tmp/outputs', params_score, restraints)
        except Exception:
            new_ligand_no_placeholders = replace_placeholders(new_ligand)
            new_score = get_biased_ligand_score(Chem.MolToSmiles(new_ligand_no_placeholders), protein, '/tmp/outputs', params_score, restraints)
        if new_score > previous_score:
            lig = new_ligand
            previous_score = new_score
            l = len(placeholder_dict)
            while f'Bond{l}' in placeholder_dict:
                l += 1
            placeholder_dict[f'Bond{l}'] = {}
            placeholder_dict[f'Bond{l}']['placeholder_type1'] = placeholder_type
            placeholder_dict[f'Bond{l}']['placeholder_type2'] = frag_placeholder_type
            lig.GetAtomWithIdx(prop_idx[0]).SetProp(f'Bond{l}', 'Atom1')
            lig.GetAtomWithIdx(prop_idx[1]).SetProp(f'Bond{l}', 'Atom2')
        else:
            probability = np.exp((new_score-previous_score)*prob)
            random.seed()
            rand = random.random()
            if rand < probability:
                lig = new_ligand
                previous_score = new_score
                l = len(placeholder_dict)
                while f'Bond{l}' in placeholder_dict:
                    l += 1
                placeholder_dict[f'Bond{l}'] = {}
                placeholder_dict[f'Bond{l}']['placeholder_type1'] = placeholder_type
                placeholder_dict[f'Bond{l}']['placeholder_type2'] = frag_placeholder_type
                lig.GetAtomWithIdx(prop_idx[0]).SetProp(f'Bond{l}', 'Atom1')
                lig.GetAtomWithIdx(prop_idx[1]).SetProp(f'Bond{l}', 'Atom2')
        return lig, previous_score, placeholder_dict
    else:
        return lig, previous_score, placeholder_dict


def step_remove_frag(lig, frag_lib, previous_score, protein, placeholder_dict, prob, params_score, restraints):
    l = len(placeholder_dict)
    bond_numbers = [i for i in range(l)]
    random.seed()
    random.shuffle(bond_numbers)
    s = False
    removed_bond = 0
    for bond in bond_numbers:
        try:
            new = remove_frags(lig, bond, placeholder_dict)
            Chem.SanitizeMol(new, catchErrors=True)
            new_ligand_no_placeholders = Chem.MolToSmiles(remove_placeholders(new))
            if not new_ligand_no_placeholders:
                raise ValueError
            s = True
            removed_bond = bond
            break
        except Exception as e:
            continue
    if s:
        try:
            new_ligand_no_placeholders = remove_placeholders(new)
            new_score = get_biased_ligand_score(Chem.MolToSmiles(new_ligand_no_placeholders), protein, '/tmp/outputs', params_score, restraints)
        except Exception:
            new_ligand_no_placeholders = replace_placeholders(new)
            new_score = get_biased_ligand_score(Chem.MolToSmiles(new_ligand_no_placeholders), protein, '/tmp/outputs', params_score, restraints)
        if new_score > previous_score:
            lig = new
            previous_score = new_score
            removed_bonds = []
            for i in lig.GetAtoms():
                i.ClearProp(f'Bond{removed_bond}')
            for s in placeholder_dict:
                bond_atom_idxs = []
                for i in lig.GetAtoms():
                    if i.HasProp(str(s)):
                        bond_atom_idxs.append(i.GetIdx())
                if not bond_atom_idxs:
                    removed_bonds.append(s)
            for bond in removed_bonds:
                placeholder_dict.pop(bond)
        else:
            probability = np.exp((new_score-previous_score)*prob)
            random.seed()
            rand = random.random()
            if rand < probability:
                lig = new
                previous_score = new_score
                removed_bonds = []
                for i in lig.GetAtoms():
                    i.ClearProp(f'Bond{removed_bond}')
                for s in placeholder_dict:
                    bond_atom_idxs = []
                    for i in lig.GetAtoms():
                        if i.HasProp(str(s)):
                            bond_atom_idxs.append(i.GetIdx())
                    if not bond_atom_idxs:
                        removed_bonds.append(s)
                for bond in removed_bonds:
                    placeholder_dict.pop(bond)
        return lig, previous_score, placeholder_dict
    else:
        return lig, previous_score, placeholder_dict
