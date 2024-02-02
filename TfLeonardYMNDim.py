from __future__ import annotations

import functools

import numpy as np
import tensorflow as tf
import copy

from approximations import get_square_root_polynomial_approximation_coefficients

import lattice as lt

import argparse
import time

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(1)


# ************************************************
# A random gauge transformation is a useful tool to check
# the correct implementation of the measurements and of the
# gauge action. As the theory should be gauge invariant, 
# we must get the same result before and after a gauge
# transformation
# ************************************************

def random_gauge_transformation(cfg: lt.Configuration):
    '''Perform a random gauge rotation of the link field
    in the configuration cfg'''
    volume = lookup_tables.local_volume

    transform = tf.complex(
        real=tf.random.uniform(minval=-1, maxval=1, shape=(volume, cfg.colors, cfg.colors), dtype=tf.float64),
        imag=tf.random.uniform(minval=-1, maxval=1, shape=(volume, cfg.colors, cfg.colors), dtype=tf.float64))

    q, _ = tf.linalg.qr(transform)
    det = tf.linalg.det(q)
    q = tf.convert_to_tensor(
        [[q[:, i, j] / (det ** (1. / cfg.colors)) for j in range(cfg.colors)] for i in range(cfg.colors)])

    rotated_gauge_field = []
    for mu in range(cfg.number_of_dimensions):
        rotated_gauge_field.append(
            np.einsum("ikc,klc,jlc->ijc", q, cfg.gauge_field[mu], tf.math.conj(translate(q, mu, +1)))
        )

    return lt.Configuration(gauge_field=tf.convert_to_tensor(rotated_gauge_field),
                            geometry=cfg.geometry,
                            colors=cfg.colors)


import lattice
import os


def save_configuration(config: lattice.Configuration, beta, kappa, cfg_number, output_folder=""):
    geometry_str = "_".join(str(elem) for elem in config.geometry)
    beta_str = str(beta).replace(".", "p")
    kappa_str = str(kappa).replace(".", "p")
    output_folder += f"lattice_{beta_str}b_{kappa_str}k_{geometry_str}/"

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    with open(output_folder + f'lattice_{beta_str}b_{kappa_str}k_{geometry_str}_{cfg_number}.cfg', 'wb') as handle:
        pickle.dump(config, handle)


def save_measurements(measurements, geometry, beta, kappa, cfg_number, output_folder=""):
    geometry_str = "_".join(str(elem) for elem in geometry)
    beta_str = str(beta).replace(".", "p")
    kappa_str = str(kappa).replace(".", "p")
    output_folder += f"lattice_{beta_str}b_{kappa_str}k_{geometry_str}/"

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    with open(output_folder + f'lattice_{beta_str}b_{kappa_str}k_{geometry_str}_{cfg_number}.measurement.cfg',
              'wb') as handle:
        pickle.dump(measurements, handle)


def load_configuration(geometry, beta, kappa, cfg_number: int = None, output_folder=""):
    geometry_str = "_".join(str(elem) for elem in geometry)
    beta_str = str(beta).replace(".", "p")
    kappa_str = str(kappa).replace(".", "p")
    output_folder += f"lattice_{beta_str}b_{kappa_str}k_{geometry_str}/"

    if cfg_number is None or cfg_number < 0:
        cfg_number = 1
        while True:
            if not os.path.exists(
                    output_folder + f'lattice_{beta_str}b_{kappa_str}k_{geometry_str}_{cfg_number + 1}.cfg'):
                break
            cfg_number += 1

    print("Trying to load configuration number", cfg_number)
    geometry_str = "_".join(str(elem) for elem in geometry)
    beta_str = str(beta).replace(".", "p")
    kappa_str = str(kappa).replace(".", "p")
    with open(output_folder + f'lattice_{beta_str}b_{kappa_str}k_{geometry_str}_{cfg_number}.cfg', 'rb') as handle:
        return pickle.load(handle), cfg_number


parser = argparse.ArgumentParser(description='TfLeonardYM.')

parser.add_argument('--geometry',
                    metavar='Lx,Ly,Lz,Lt',
                    type=int,
                    nargs='+',
                    help='The geometry of lattice')

parser.add_argument('--beta',
                    metavar='B',
                    type=float,
                    help='The inverse squared gauge coupling')

parser.add_argument('--kappa',
                    metavar='K',
                    type=float,
                    help='The hopping parameter for the dirac-wilson operator')

parser.add_argument('--mass',
                    metavar='M',
                    type=float,
                    help='The hopping parameter for the dirac-wilson operator',
                    default=0.0)

parser.add_argument('-mpi_grid',
                    metavar='Px,Py,Pz,Pt',
                    type=int,
                    nargs='+',
                    help='The MPI grid of used to split the lattice',
                    default=[1, 1, 1, 1])

parser.add_argument('-mpi',
                    metavar='True/False',
                    type=bool,
                    nargs='+',
                    help='Use MPI?',
                    default=False)

parser.add_argument('-colors',
                    metavar='Nc',
                    type=int,
                    help='Number of colors of the gauge theory',
                    default=3)

parser.add_argument('-number_of_configurations',
                    metavar='N',
                    type=int,
                    help='Number of configurations to be generated',
                    default=200)

parser.add_argument('-starting_configuration_number',
                    metavar='Nconf',
                    type=int,
                    help='Configuration number to (re)start measurements and simulations',
                    default=-1)

parser.add_argument('-measurement_only',
                    metavar='meas',
                    type=bool,
                    help='If true, only measurements will be performed',
                    default=False)

parser.add_argument('-t_hmc',
                    metavar='t',
                    type=float,
                    help='The trajectory length for HMC',
                    default=1.0)

parser.add_argument('-integration_steps',
                    metavar='s',
                    nargs='+',
                    type=int,
                    help='The number of integration steps for the HMC trajectory',
                    default=[6])

parser.add_argument('-number_of_dimensions',
                    metavar='s',
                    type=int,
                    help='The number of space-time dimensions',
                    default=4)

parser.add_argument('-output_folder',
                    metavar="outf",
                    type=str,
                    help='The output folder where the measurements and the configurations are stored',
                    default="")

parser.add_argument('-boundary_conditions',
                    metavar="bc",
                    type=str,
                    help='The boundary conditions',
                    default="periodic")

args = parser.parse_args()

# ************************************************
# Define the lattice geometry and compute the lookup tables
# in the beginning of the Monte Carlo simulation.
# ************************************************

geometry = tuple(args.geometry)
mpi_grid = tuple(args.mpi_grid)
output_folder = args.output_folder

import metropolis

metropolis.set_metropolis(args.mpi)

import translate

translate.set_translate(args.mpi, args.geometry, mpi_grid)

import boundary_conditions

boundary_conditions.set_antiperiodic_field_in_t_direction(args.geometry, -1)

boundary_condition_provider = None
if args.boundary_conditions == "periodic":
    print("Periodic fermion boundary conditions")


    def f(x):
        return x


    boundary_condition_provider = f
elif args.boundary_conditions == "time-antiperiodic":
    print("Using antiperiodic fermion boundary conditions in the time direction")


    def f(x):
        return boundary_conditions.apply_antiboundary_conditions_in_t_direction(x)


    boundary_condition_provider = f


# ************************************************
# In MPI mode the output is printed only by the master rank
# ************************************************

def MPI_output(*args, **kwars):
    from mpi4py import MPI
    '''Output only for the master rank'''
    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()

    if my_rank == 0:
        __builtins__.print(*args, **kwars)


if args.mpi:
    print = MPI_output

import hmc

# Initialize the configuration
config = hmc.hotstart(geometry, args.colors, args.number_of_dimensions)

import dirac
import fermion_action
import gauge_action
import fermion_measurements
import measurements

#### HMC

import dirac_solver
import pickle
from translate import lookup_tables, translate, global_sum

try:
    config, cfg_number = load_configuration(geometry,
                                            args.beta,
                                            args.kappa,
                                            args.starting_configuration_number,
                                            output_folder)
    print("Configuration loaded!")
except Exception as e:
    print(f"Failed to load config: {e}")
    print("Hotstart of a new gauge field")
    cfg_number = 0

multishift_solver = dirac_solver.multishift_solver(70000, 1e-11)
dirac_wilson_operator = dirac.DiracWilsonOperator(config,
                                                  args.kappa,
                                                  True,
                                                  representation="adjoint",
                                                  boundary_condition_provider=boundary_condition_provider)

square_root_polynomial_approximation_coefficients = get_square_root_polynomial_approximation_coefficients(40)
square_root_polynomial_approximation = dirac.PolynomialApproximation(
    dirac.SquareOperator(dirac_wilson_operator),
    roots=square_root_polynomial_approximation_coefficients[1:],
    scaling=square_root_polynomial_approximation_coefficients[0])

overlap = dirac.Overlap(dirac_wilson_operator, square_root_polynomial_approximation, args.mass, True)

dirac_operator = dirac_wilson_operator

from approximations import (force_level1_rational_approximation_coefficients,
                            heatbath_rational_approximation_coefficients,
                            metropolis_rational_approximation_coefficients)

force_rational_approximation = dirac.RationalApproximation(
    dirac.SquareOperator(dirac_operator),
    force_level1_rational_approximation_coefficients[:len(force_level1_rational_approximation_coefficients) // 2],
    force_level1_rational_approximation_coefficients[len(force_level1_rational_approximation_coefficients) // 2:],
    shift=0,
    solver=multishift_solver
)

heatbath_rational_approximation = dirac.RationalApproximation(
    dirac.SquareOperator(dirac_operator),
    heatbath_rational_approximation_coefficients[:len(heatbath_rational_approximation_coefficients) // 2],
    heatbath_rational_approximation_coefficients[len(heatbath_rational_approximation_coefficients) // 2:],
    shift=0,
    solver=multishift_solver
)

metropolis_rational_approximation = dirac.RationalApproximation(
    dirac.SquareOperator(dirac_operator),
    metropolis_rational_approximation_coefficients[:len(metropolis_rational_approximation_coefficients) // 2],
    metropolis_rational_approximation_coefficients[len(metropolis_rational_approximation_coefficients) // 2:],
    shift=0,
    solver=multishift_solver
)

number_pseudofermions = 2

gluino_action = fermion_action.n_flavor(
    [copy.deepcopy(force_rational_approximation) for _ in range(number_pseudofermions)],
    [copy.deepcopy(metropolis_rational_approximation) for _ in range(number_pseudofermions)],
    [copy.deepcopy(heatbath_rational_approximation) for _ in range(number_pseudofermions)],
    multishift_solver,
    dirac_operator)

print("Number of flavors:", gluino_action.number_of_flavors())

actions = [gluino_action, gauge_action.yang_mills(config, args.beta)]

if not args.measurement_only:
    print("Using beta", args.beta)
    print("Using kappa", args.kappa)
    # Run the simulation
    for i in range(cfg_number + 1, cfg_number + args.number_of_configurations + 1):
        start = time.time()
        pstart = time.process_time()

        random_vectors = [
            hmc.generate_random_vector(args.colors, dirac_operator.spinor_dimension, representation="adjoint",
                                       stddev=0.5)
            for _ in range(number_pseudofermions)]

        gluino_action.initialize_pseudofermions(random_vectors)

        # rev = hmc.reversibility_check(config,
        #                               actions=actions,
        #                                steps=args.integration_steps,
        #                                delta_t=args.t_hmc)
        # print("Rev:", rev)

        config, dE, acc = hmc.hybrid_mc(config,
                                        actions=actions,
                                        steps=args.integration_steps,
                                        delta_t=args.t_hmc)

        save_configuration(config, args.beta, args.kappa, i, output_folder)

        dirac_operator.set_gauge_configuration(config)

        config_measurements = [
            measurements.plaquette(config),
            fermion_measurements.chiral_condensate(dirac_operator),
            fermion_measurements.pion_correlator(dirac_operator, 1, lookup_tables, global_sum)
        ]
        for measurement in config_measurements:
            print(measurement.name, "for configuration", i, ":", measurement.value)

        save_measurements(config_measurements, config.geometry, args.beta, args.kappa, i, output_folder)

        end = time.time()
        pend = time.process_time()
        print("Time elapsed: ", end - start, " seconds")
        print("Process time elapsed: ", pend - pstart, " seconds")

# print(multi_smearing_operators(config, [lambda x: polyakov_operators(x)], skip = 3, no_smear_direction = 4))
# print(multi_smearing_operators(random_gauge_transformation(config), [lambda x: polyakov_operators(x)], skip = 3, no_smear_direction = 4))
# python ./TfLeonardYMNDim.py --geometry 20 8 --beta 3.0 --kappa 0.15 -number_of_dimensions=2 -colors=2 -integration_steps 2 3 2
# python ./TfLeonardYMNDim.py --geometry 24 16 --beta 3.0 --kappa -0.15 -number_of_dimensions=2 -colors=2 -integration_steps 4 3 3
