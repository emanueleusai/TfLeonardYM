import dirac
import tensorflow as tf
import lattice as lt
import dirac_solver
import numpy as np
import algebra_utils
import lie_generators
from measurements import Measurement


def generate_random_z2_noise(volume, nc=3, spinor_dimension=4, representation="fundamental"):
    if representation == "fundamental":
        return (tf.cast(tf.random.uniform(shape=(spinor_dimension, nc, volume), minval=0, maxval=2, dtype=tf.int32),
                        dtype=tf.complex128) - 0.5) * 2

    elif representation == "adjoint":
        generators = lie_generators.generators(nc, "fundamental")

        vector = (tf.cast(
            tf.random.uniform(shape=(spinor_dimension, nc * nc - 1, volume), minval=0, maxval=2, dtype=tf.int32),
            dtype=tf.complex128) - 0.5) * 2

        return tf.einsum("mic,ijk->mjkc", vector, generators)


def generate_point_source(site, color, spin, volume, nc=3, spinor_dimension=4, representation="fundamental"):
    if representation == "fundamental":
        mask = np.zeros(shape=(spinor_dimension, nc, volume), dtype=np.float64)
        mask[spin, color, site] = 1

        return tf.convert_to_tensor(mask)
    elif representation == "adjoint":
        generators = lie_generators.generators(nc, "fundamental")
        mask = np.zeros(shape=(spinor_dimension, nc * nc - 1, volume), dtype=np.complex128)
        mask[spin, color, site] = 1
        vector = tf.convert_to_tensor(mask)

        return tf.einsum("mic,ijk->mjkc", vector, generators)


def pion_correlator(dirac_operator: dirac.Operator,
                    correlator_direction,
                    lookup_tables,
                    global_summator) -> Measurement:
    coordinates = lookup_tables.global_coordinates
    propagator = dirac.Propagator(
        dirac_operator,
        dirac_solver.biconjugate_gradient(5000, 1e-13))

    sources = []
    solutions = []

    nc = dirac_operator.number_of_colors
    if dirac_operator.representation == "adjoint":
        nc = dirac_operator.number_of_colors ** 2 - 1

    for spin in range(dirac_operator.spinor_dimension):
        for c in range(nc):
            sources.append(generate_point_source(0,
                                                 c,
                                                 spin,
                                                 dirac_operator.volume,
                                                 dirac_operator.number_of_colors,
                                                 dirac_operator.spinor_dimension,
                                                 dirac_operator.representation))

    for source in sources:
        solutions.append(propagator.apply(source))

    correlator = np.zeros(shape=dirac_operator.cfg.geometry[correlator_direction], dtype=np.complex128)

    for solution in solutions:
        if dirac_operator.representation == "fundamental":
            dot = tf.einsum("icm,icm->m", tf.math.conj(solution), solution)
        elif dirac_operator.representation == "adjoint":
            dot = tf.einsum("icdm,idcm->m", tf.math.conj(solution), solution)

        for z in range(dirac_operator.cfg.geometry[correlator_direction]):
            timeslice = tf.where(coordinates[correlator_direction] == z)

            if len(timeslice) != 0:
                correlator[z] += tf.reduce_sum(tf.gather(dot, timeslice))

    # Gather the sum from all nodes
    for i in range(dirac_operator.cfg.geometry[correlator_direction]):
        correlator[i] = global_summator(correlator[i])

    return Measurement(name="pion_operator",
                       value=np.array(correlator),
                       geometry=dirac_operator.cfg.geometry,
                       colors=dirac_operator.cfg.colors)


def chiral_condensate(dirac_operator: dirac.Operator, num_estimators=20) -> Measurement:
    propagator = dirac.Propagator(
        dirac_operator,
        dirac_solver.biconjugate_gradient(5000, 1e-10))

    sources = []
    solutions = []
    for _ in range(num_estimators):
        source = generate_random_z2_noise(dirac_operator.volume,
                                          dirac_operator.number_of_colors,
                                          dirac_operator.spinor_dimension,
                                          dirac_operator.representation)
        solution = propagator.apply(source)
        sources.append(source)
        solutions.append(solution)

    return Measurement(name="chiral_condensate",
                       value=np.mean(
                           [algebra_utils.dot(source, solution) / dirac_operator.volume for source, solution in
                            zip(sources, solutions)]),
                       geometry=dirac_operator.cfg.geometry,
                       colors=dirac_operator.cfg.colors)
