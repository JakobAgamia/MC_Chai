from rdkit import Chem
from score_functions import get_biased_ligand_score
from score_functions import get_dice_score
from main_functions import step
import os
from utility_functions import save_structurs


class MonteCarloMethode:
    def __init__(self, protein, ligand, name, first_step=0):
        self.protein = protein
        self.ligand = ligand
        self.name = name
        self.first_step = first_step
        with open(self.name, "w") as f:
            pass

    def monte_carlo_full(self, num_steps, max_len, params=None, prob=50, save=False, params_score=None, restraints=False, file_path=None):
        if file_path is None:
            file_path = self.name
        if params is None:
            params = [0.7, 0.6, 0.55, 0.525, 0.5, 0.45, 0.3, 0.2]
        if params_score is None:
            params_score = [0.8, 0.1, 0.05, 0.05]
        lig = Chem.MolFromSmiles(self.ligand)
        score = get_biased_ligand_score(self.ligand, self.protein, '/tmp/outputs', params_score, restraints)
        lig_score_list = [[self.ligand, score]]
        if save:
            name = os.path.splitext(file_path)[0]
            steps = 0
            save_structurs(name, steps)
        log_name = os.path.splitext(file_path)[0] + "_log.txt"
        with open(log_name, 'w') as test_sim:
            test_sim.write(f'Molecule: {self.name} \n')
        if self.first_step == 0:
            with open(file_path, 'w') as test_sim:
                test_sim.write(f'\n Starting structure: {Chem.MolToSmiles(lig)} \n')
                self.first_step += 1
        else:
            with open(file_path, 'a') as test_sim:
                test_sim.write(f'\n Starting structure: {Chem.MolToSmiles(lig)} \n')
        with open(file_path, 'a') as test_sim:
            test_sim.write(f' Params: {params}\n')
            test_sim.write(f' Params score: {params_score}\n')
            test_sim.write(f' N: {prob}\n')
        best_score = score
        best_lig = lig
        st = 0
        s = 0
        count = 0
        for i in range(num_steps):
            st += 1
            new = step(self.protein, lig, score, max_len, params, prob, log_name, params_score, restraints, st, lig_score_list)
            lig = new[0]
            score = new[1]
            if score > best_score:
                best_score = score
                best_lig = lig
            if score > best_score:
                best_score = score
                best_lig = lig
            s += 1
            if save:
                if s % 1 == 0:
                    name = os.path.splitext(file_path)[0]
                    steps = s + count
                    save_structurs(name, steps)
            if s == 100:
                s = 0
                count += 100
                lig = best_lig
                score = best_score
                if count == 100:
                    with open(file_path, 'a') as test_sim:
                        test_sim.write(f'\n Steps: {count} \n Ligand:{Chem.MolToSmiles(lig)} Score:\n {score} \n')
                else:
                    with open(file_path, 'a') as test_sim:
                        test_sim.write(f'\n Steps: {count} \n Ligand:{Chem.MolToSmiles(lig)} \n Score:{score} \n')
        with open(file_path, 'a') as test_sim:
            test_sim.write(f'\n Final results:\n Steps: {count} \n Ligand:{Chem.MolToSmiles(lig)} \n Score:{score} \n')
#        self.ligand = Chem.MolToSmiles(best_lig)
        return best_score, Chem.MolToSmiles(best_lig)
