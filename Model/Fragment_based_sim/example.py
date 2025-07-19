from Monte_carlo import MonteCarloMethode
from rdkit import Chem
from utility_functions import read_smiles_with_placeholders


protein_bromo = ['SMNPPPPETSNPNKPKRQTNQLQYLLRVVLKTLWKHQFAWPFQQPVDAVKLNLPDYYKIIKTPMDMGTIKKRLENNYYWNAQECIQDFNTMFTNCYIYNKPGDDIVLMAEALEKLFLQKINELPTEE ']
protein_serine = ['MAHHHHHHLEVLFQGPLLSKINSLAHLRAAPCNDLHATKLAPGKEKEPLESQYQVGPLLGSGGFGSVYSGIRVSDNLPVAIKHVEKDRISDWGELPNGTRVPMEVVLL'
           'KKVSSGFSGVIRLLDWFERPDSFVLILERPEPVQDLFDFITERGALQEELARSFFWQVLEAVRHCHNCGVLHRDIKDENILIDLNRGELKLIDFGSGALLKDTVYTDF'
           'DGTRVYSPPEWIRYHRYHGRSAAVWSLGILLYDMVCGDIPFEHDEEIIRGQVFFRQRVSSECQHLIRWCLALRPSDRPTFEEIQNHPWMQDVLLPQETAEIHLHSLSPGPSK']
protein_p38 = ['GSHMLEMSQERPTFYRQELNKTIWEVPERYQNLSPVGSGAYGSVCAAFDTKTGLRVAVKKLSRPFQSIIHAKRTYRELRLLKHMKHENVIGLLDVFTPARSLEEFNDV'
           'YLVTHLMGADLNNIVKCQKLTDDHVQFLIYQILRGLKYIHSADIIHRDLKPSNLAVNEDSELKILDFGLARHTDDEMTGYVATRWYRAPEIMLNWMHYNQTVDIWSVG'
           'CIMAELLTGRTLFPGTDHIDQLKLILRLVGTPGAELLKKISSESARNYIQSLTQMPKMNFANVFIGANPLAVDLLEKMLVLDSDKRITAAQALAHAYFAQYHDPDDEP']
protein = protein_bromo


frag_lib = read_smiles_with_placeholders('./bric_list.txt')
ligand = Chem.MolToSmiles(frag_lib[0])
example = MonteCarloMethode(protein, ligand, f'./random_frags.txt', frag_lib)
result = example.monte_carlo_full(1000, save=True)
# inputs: num_steps: number of simulation steps
#         params(optional): a list defining the likelihood of adding/removing fragments: 0.4
#         prob(optional): the beta factor to decide the probability of worse structures being accepted
#         save(optional): if set to true the cif structure files will be saved in s folder
#         params_score(optional): parameters to determine how strongly the chai-1 score and the biasing scores are taken into account, example: [0.8, 0.1, 0.05, 0.05]
#         restraints(optional): if set top true , chai-1 will expect a file pocket.restraints which should define pocket restraint check chai-1 documentation for how to set up
