from rdkit import Chem
import random
from score_functions import get_biased_ligand_score
from score_functions import get_dice_score
from main_functions import step_combine
from main_functions import step_remove_frag
from utility_functions import identify_placeholders
from utility_functions import replace_placeholders
from utility_functions import remove_placeholders
from utility_functions import save_structurs
import os
import copy


class MonteCarloMethode:
    def __init__(self, protein, ligand, name, frag_lib, first_step=0):
        self.protein = protein
        self.ligand = ligand
        self.name = name
        self.first_step = first_step
        self.frag_lib = frag_lib

    def monte_carlo_full(self, num_steps, params=None, prob=5, params_score=None, save=False, restraints=False):
        if params is None:
            params = 0.4
        if params_score is None:
            params_score = [0.8, 0.1, 0.05, 0.05]
        file_path = self.name
        log_name = os.path.splitext(file_path)[0] + "_log.txt"
        with open(log_name, 'w') as test_sim:
            test_sim.write(f'Molecule: {self.name} \n')
        lig = Chem.MolFromSmiles(self.ligand)
        try:
            new_ligand_no_placeholders = remove_placeholders(lig)
            score = get_biased_ligand_score(Chem.MolToSmiles(new_ligand_no_placeholders), self.protein, '/tmp/outputs', params_score, restraints)
        except Exception:
            new_ligand_no_placeholders = replace_placeholders(lig)
            score = get_biased_ligand_score(Chem.MolToSmiles(new_ligand_no_placeholders), self.protein, '/tmp/outputs', params_score, restraints)
        if self.first_step == 0:
            with open(f'{self.name}', 'w') as test_sim:
                test_sim.write(f'\n Starting structure: {Chem.MolToSmiles(lig)} \n N: {prob} \n Prob add remove: {params} \n')
                self.first_step += 1
        else:
            with open(f'{self.name}', 'a') as test_sim:
                test_sim.write(f'\n Starting structure: {Chem.MolToSmiles(lig)} \n N: {prob} \n Prob add remove: {params} \n')
        best_score = score
        best_lig = lig
        s = 0
        count = 0
        l = 1
        placeholder_dict = {}
        best_placeholder_dict = {}
        for i in range(num_steps):
            random.seed()
            operation = random.random()
            len = lig.GetNumAtoms()
            if identify_placeholders(lig):
                if operation >= params:
                    with open(log_name, 'a') as test_sim:
                        test_sim.write(f'Step: add fragment, Previous ligand: {Chem.MolToSmiles(lig)}')
                    new = step_combine(lig, self.frag_lib, score, self.protein, placeholder_dict, prob, params_score, restraints)
                    lig = new[0]
                    score = new[1]
                    placeholder_dict = new[2]
                    with open(log_name, 'a') as test_sim:
                        test_sim.write(f', New ligand: {Chem.MolToSmiles(lig)}, Score: {score} \n')
                    if score > best_score:
                        best_score = score
                        best_lig = lig
                        best_placeholder_dict = copy.deepcopy(placeholder_dict)
                else:

                    with open(log_name, 'a') as test_sim:
                        test_sim.write(f'Step: remove fragment, Previous ligand: {Chem.MolToSmiles(lig)}')
                    new = step_remove_frag(lig, self.frag_lib, score, self.protein, placeholder_dict, prob, params_score, restraints)
                    lig = new[0]
                    score = new[1]
                    placeholder_dict = new[2]
                    with open(log_name, 'a') as test_sim:
                        test_sim.write(f', New ligand: {Chem.MolToSmiles(lig)}, Score: {score} \n')
                    if score > best_score:
                        best_score = score
                        best_lig = lig
                        best_placeholder_dict = copy.deepcopy(placeholder_dict)
            else:
                with open(log_name, 'a') as test_sim:
                    test_sim.write(f'Step: remove fragment, Previous ligand: {Chem.MolToSmiles(lig)}')
                new = step_remove_frag(lig, self.frag_lib, score, self.protein, placeholder_dict, prob, params_score, restraints)
                lig = new[0]
                score = new[1]
                placeholder_dict = new[2]
                with open(log_name, 'a') as test_sim:
                    test_sim.write(f', New ligand: {Chem.MolToSmiles(lig)}, Score: {score} \n')
                if score > best_score:
                    best_score = score
                    best_lig = lig
                    best_placeholder_dict = copy.deepcopy(placeholder_dict)
            s += 1
            if save:
                if s % 1 == 0:
                    name = os.path.splitext(file_path)[0]
                    steps = s + count
                    save_structurs(name, steps)
            if s == 50:
                s = 0
                lig = best_lig
                score = best_score
                placeholder_dict = copy.deepcopy(best_placeholder_dict)
            if s % 50 == 0:
                count += 50
                with open(f'{self.name}', 'a') as test_sim:
                    test_sim.write(f'\n Steps: {count} \n {Chem.MolToSmiles(lig)} \n Score: {score} ) \n Ligand:{Chem.MolToSmiles(best_lig)}')
        with open(f'{self.name}', 'a') as test_sim:
            test_sim.write(f'\n Final Result: \n  Steps: {count} \n {Chem.MolToSmiles(lig)} \n Score: {score} ) \n Ligand:{Chem.MolToSmiles(best_lig)}')
        self.ligand = Chem.MolToSmiles(best_lig)
        return best_score, best_lig, best_placeholder_dict
