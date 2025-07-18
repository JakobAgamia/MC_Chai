from rdkit import Chem
from pathlib import Path
import numpy as np
import torch
from chai_lab.chai1 import run_inference
import random
from rdkit.Chem import AllChem, DataStructs
import time
from rdkit.Chem.rdShapeHelpers import ShapeTanimotoDist
from rdkit.Contrib.SA_Score import sascorer
from rdkit.Chem import QED
from rdkit.Chem import Descriptors, Crippen, Lipinski
import shutil


def get_ligand_score(ligand_name, protein_names, out_dir, restraints):
    if not restraints:
        if torch.cuda.device_count() > 0:
            tries = 0
            while tries < 20:
                try:
                    sequences = []
                    for i in range(len(protein_names)):
                        sequences.append((f"protein|name=protein{i}", f"{protein_names[i]}"))
                    sequences.append(("ligand|name=example-ligand-as-smiles", f"{ligand_name}"))
                    fasta_path = Path("/tmp/example.fasta")
                    with open(fasta_path, "w") as fasta_file:
                        for seq_id, seq_data in sequences:
                            fasta_file.write(f">{seq_id} \n{seq_data.strip()}\n")
                    output_dir = Path(out_dir)
                    if output_dir.exists():
                        shutil.rmtree(output_dir)
                    output_dir.mkdir(exist_ok=True)

                    candidates = run_inference(
                        fasta_file=fasta_path,
                        output_dir=output_dir,
                        num_trunk_recycles=3,
                        num_diffn_timesteps=50,
                        seed=42,
                        device=torch.device("cuda"),
                        use_esm_embeddings=True,
                    )
                    scores = [rd.aggregate_score for rd in candidates.ranking_data]
                    i = 0
                    for s in scores:
                        if s > i:
                            i = s
                    return i
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print("CUDA out of memory error. Pausing and retrying...")
                        torch.cuda.empty_cache()
                        time.sleep(10)
                        tries += 1
                    else:
                        raise e
        else:
            return random.random()
    else:
        if torch.cuda.device_count() > 0:
            tries = 0
            while tries < 20:
                try:
                    sequences = [(f"protein|name=protein{1}", f"{protein_names[0]}"),
                                 ("ligand|name=example-ligand-as-smiles", f"{ligand_name}"),
                                 (f"protein|name=protein{0}", f"{protein_names[1]}")]
                    fasta_path = Path("/tmp/example.fasta")
                    with open(fasta_path, "w") as fasta_file:
                        for seq_id, seq_data in sequences:
                            fasta_file.write(f">{seq_id} \n{seq_data.strip()}\n")
                    output_dir = Path(out_dir)
                    if output_dir.exists():
                        shutil.rmtree(output_dir)
                    output_dir.mkdir(exist_ok=True)

                    candidates = run_inference(
                        fasta_file=fasta_path,
                        output_dir=output_dir,
                        constraint_path=Path(__file__).with_name("pocket.restraints"),
                        num_trunk_recycles=3,
                        num_diffn_timesteps=50,
                        seed=42,
                        device=torch.device("cuda"),
                        use_esm_embeddings=True,
                    )
                    cif_paths = candidates.cif_paths
                    scores = [rd.aggregate_score for rd in candidates.ranking_data]
                    i = 0
                    for s in scores:
                        if s > i:
                            i = s
                    return i
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print("CUDA out of memory error. Pausing and retrying...")
                        torch.cuda.empty_cache()
                        time.sleep(10)
                        tries += 1
                    else:
                        raise e
        else:
            return random.random()


def get_dice_score(lig1, lig2):
    fp1 = AllChem.GetMorganFingerprintAsBitVect(lig1, 2, nBits=1024)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(lig2, 2, nBits=1024)
    dice_similarity = DataStructs.DiceSimilarity(fp1, fp2)
    return dice_similarity


def get_tanimoto_dist(ref_mol, candidate_mol):
    ref_mol = Chem.AddHs(ref_mol)
    AllChem.EmbedMolecule(ref_mol, randomSeed=42)
    AllChem.UFFOptimizeMolecule(ref_mol)
    candidate_mol = Chem.AddHs(candidate_mol)
    AllChem.EmbedMolecule(candidate_mol, randomSeed=42)
    AllChem.UFFOptimizeMolecule(candidate_mol)
    similarity = 1 - ShapeTanimotoDist(ref_mol, candidate_mol)
    return similarity


def calc_esol(mol):
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    rotors = Lipinski.NumRotatableBonds(mol)
    a_count = 0
    for i in mol.GetAtoms():
        if i.GetIsAromatic():
            a_count += 1
    ap = a_count/mol.GetNumAtoms()
    return 0.16 - 0.63 * logp + 0.066 * rotors -0.74 * ap


def calc_esol_unmodyfied(mol):
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    rotors = Lipinski.NumRotatableBonds(mol)
    a_count = 0
    for i in mol.GetAtoms():
        if i.GetIsAromatic():
            a_count += 1
    ap = a_count/mol.GetNumAtoms()
    return 0.16 - 0.63 * logp + 0.066 * rotors -0.74 * ap -0.0062*mw


def get_biased_ligand_score(ligand_name, protein, out_dir, params_score, restraints):
    lig = Chem.MolFromSmiles(ligand_name)
    complex_score = get_ligand_score(ligand_name, protein, out_dir, restraints)
    #complex_score = random.random()
    sa_score = sascorer.calculateScore(lig)
    if sa_score < 5.0:
        sa_score = 1
    else:
        sa_score = (10 - sascorer.calculateScore(lig))/9
    qed_score = QED.qed(lig, w=(0.0, 0.46, 0.05, 0.61, 0.06, 0.65, 0.48, 0.95))
    esol = calc_esol(lig)
    esol_score = (esol + 10)/10
    score = params_score[0] * complex_score + params_score[1] * sa_score + params_score[2] * qed_score + params_score[3] * esol_score
    return score
