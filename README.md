# About   
This repository presents an approach for ligand discovery for protein bindign pockets, by combining Monte Carlo (MC) simulations with the model Chai-1 
([Chai-1 github](https://github.com/chaidiscovery/chai-lab), [Chai-1 technical report](https://www.biorxiv.org/content/10.1101/2024.10.10.615955v2)). 
There are two types of simulations presented here:   
- The basic simulation explores chemical space by making small changes to a simple starting strucure and evaluating new strcutres with Chai-1's confidence score.
- The fragment based simulation recombines molecular fragments to build up the ligand. 
## Requirements
To run the model code chai-lab==0.5.1 is required. Refer to ([Chai-1 github](https://github.com/chaidiscovery/chai-lab)) for installation instructions and machine requiremnets. All other neccessary packages will outomactically be
installed by the Chai-1 installation.
## Usage
Both the folder for the basic simulation and the fragment based simulation include a example.py file, which shows how to use the simulations.   


*The paper explaining this model is not yet published at this time. The data folder includes results of that paper.
