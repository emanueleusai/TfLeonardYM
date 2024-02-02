# TfLeonardYM
Tensorflow code to perform lattice simulations of Yang-Mills theories in an arbitrary number of D space-time euclidean dimensions and possibly for several different gauge groups.

## Algorithms

The code implements Monte Carlo sampling of gauge-field configuration integrating the classical equation of motion defined in D+1 dimensions (Hybrid Monte Carlo, HMC). Fermion fields are integrated out and are simulated in terms of an effective theory.

Each time a configuration is generated, several measurements can performed, such as correlators or vacuum expectation values of local operators. The ensemble average can be straightforwardly computed at the end of the simulation.

## Usage

The script requires tensorflow, and also MPI if the configurations are distributed in parallel on several machines. To run the script, you can execute for instance

``
python ./TfLeonardYMNDim.py --geometry 24 16 --beta 3.0 --kappa -0.15 -number_of_dimensions=2 -colors=2 -integration_steps 4 3 3
``

to start a simulation on a two dimension lattice of size 24x16, where the gauge fields belong to the gauge group SU(2).

Run also

``
python ./TfLeonardYMNDim.py --help
``

to see all options and their meaning.