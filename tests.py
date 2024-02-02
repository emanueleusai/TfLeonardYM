import numpy as np

colors = 2
number_of_dimensions = 2
geometry = [8, 8]
bc = "time-antiperiodic"

import translate

translate.set_translate(False, geometry, [1, 1])

import boundary_conditions

boundary_conditions.set_antiperiodic_field_in_t_direction(geometry)

if bc == "periodic":
    def f(x, _):
        return x

    boundary_condition_provider = f
elif bc == "time-antiperiodic":
    def f(x, f):
        return boundary_conditions.apply_antiboundary_conditions_in_t_direction(x, f)

    boundary_condition_provider = f


import dirac
import tensorflow as tf
from hmc import generate_random_vector, hotstart
import algebra_utils
import dirac_solver
import lattice as lt
import fermion_action
import lie_generators

# Initialize the configuration
config = hotstart(geometry, colors, number_of_dimensions)

square_root_polynomial_approximation_coefficients = np.array([1.0, 1 + 3.11j, 1 - 3.11j])

dirac_operator = dirac.DiracWilsonOperator(config,
                                           0.1,
                                           True,
                                           representation="fundamental",
                                           boundary_condition_provider=boundary_condition_provider)
random_vector = generate_random_vector(colors, dirac_operator.spinor_dimension)
random_vector_2 = generate_random_vector(colors, dirac_operator.spinor_dimension)
print(random_vector.shape)

test1 = dirac_operator.multiply_in_fundamental_representation(random_vector)
test2 = dirac_operator.simple_multiply_in_fundamental_representation(random_vector)
test3 = dirac_operator.multiply_in_fundamental_representation(random_vector_2)
print(test1[0, :, 0], "must be equal to", test2[0, :, 0], "and this should not be zero:", random_vector[0, 0, 0])
print("The following two numbers must be equal")
print(tf.math.reduce_sum(tf.einsum("mic,mic->m", tf.math.conj(random_vector_2), test1)))
print(tf.math.reduce_sum(tf.einsum("mic,mic->m", tf.math.conj(test3), random_vector)))

print("This must be zero", tf.math.reduce_sum(test1 - test2))

print("Overlap:")

square_root_polynomial_approximation = dirac.PolynomialApproximation(
    dirac.SquareOperator(dirac_operator),
    roots=square_root_polynomial_approximation_coefficients[1:],
    scaling=square_root_polynomial_approximation_coefficients[0])

overlap = dirac.Overlap(dirac_operator, square_root_polynomial_approximation, 0, True)
test1 = overlap.apply(random_vector)
test2 = overlap.apply(random_vector_2)
print("The following two numbers must be equal",
      algebra_utils.dot(test2, random_vector),
      algebra_utils.dot(random_vector_2, test1))

print("Adjoint:")
random_vector = generate_random_vector(colors, dirac_operator.spinor_dimension, representation="adjoint")
random_vector_2 = generate_random_vector(colors, dirac_operator.spinor_dimension, representation="adjoint")

dirac_operator = dirac.DiracWilsonOperator(config,
                                           0.1,
                                           True,
                                           boundary_condition_provider=boundary_condition_provider)
test1 = dirac_operator.simple_multiply_in_adjoint_representation(random_vector)
test2 = dirac_operator.multiply_in_adjoint_representation(random_vector)
rv_adj = 2 * tf.einsum("ijk,mkjc->mic", lie_generators.generators(colors, "fundamental"), random_vector)
test3 = dirac_operator.multiply_in_fundamental_representation(rv_adj)
test4 = 2 * tf.einsum("ijk,mkjc->mic", lie_generators.generators(colors, "fundamental"), test1)

print("This number must be zero", tf.einsum("mjic,mjic", tf.math.conj(test1 - test2), test1 - test2))
print("This number must be zero", tf.einsum("mjc,mjc", tf.math.conj(test3 - test4), test3 - test4))
print("The following two numbers must be equal")
print(tf.einsum("mjic,mjic", tf.math.conj(random_vector_2), test2))
test2 = dirac_operator.multiply_in_adjoint_representation(random_vector_2)
print(tf.einsum("mjic,mjic", tf.math.conj(test2), random_vector))

print("Solver:")

squared_dirac_wilson_operator = dirac.SquareOperator(dirac_operator)
multishift_solver = dirac_solver.multishift_solver(1000, 1e-11)
shifts = [0.5, 0.1, 0.01]
results = multishift_solver.solve(squared_dirac_wilson_operator, random_vector, shifts)

for i, shift in enumerate(shifts):
    shifted_dirac = dirac.ShiftedOperator(squared_dirac_wilson_operator, shift)
    test = shifted_dirac.apply(results[i])
    print("Test on shift", shift, "(it must be zero):", algebra_utils.dot(test - random_vector, test - random_vector))

solver = dirac_solver.biconjugate_gradient(1000, 1e-11)
result = solver.solve(shifted_dirac, random_vector)
print("BiCG versus MMMR test")
print(algebra_utils.dot(result - results[-1], result - results[-1]))
test = shifted_dirac.apply(result)
print(algebra_utils.dot(test - random_vector, test - random_vector))

print("Hermitian inverse squared")
result = solver.solve(squared_dirac_wilson_operator, random_vector)
print(algebra_utils.dot(random_vector_2, result))
result = solver.solve(squared_dirac_wilson_operator, random_vector_2)
print(algebra_utils.dot(result, random_vector))

print("Lie derivative")

kappa = 0.1

random_vector = generate_random_vector(colors, dirac_operator.spinor_dimension, representation="fundamental")

dirac_operator = dirac.DiracWilsonOperator(config, kappa, True, representation="fundamental")

square_root_polynomial_approximation = dirac.PolynomialApproximation(
    dirac.SquareOperator(dirac_operator),
    roots=square_root_polynomial_approximation_coefficients[1:],
    scaling=square_root_polynomial_approximation_coefficients[0])

overlap = dirac.Overlap(dirac_operator, square_root_polynomial_approximation, 0.01, hermitian=True)

force_rational_approximation = dirac.RationalApproximation(
    dirac.SquareOperator(overlap),
    [1.0],
    [0.0],
    shift=0,
    solver=multishift_solver
)

force_rational_approximation.test_inverter(random_vector)

# der = dirac_operator.lie_derivative(random_vector, random_vector_2)
print("Energy")

faction_original = fermion_action.n_flavor(
    [force_rational_approximation],
    [force_rational_approximation],
    [force_rational_approximation],
    multishift_solver,
    overlap)

faction_original.initialize_pseudofermions([random_vector])
der = faction_original.force()

for epsilon in (0.01, 0.001, 0.0001):
    ctest = config.gauge_field.numpy()
    ctest[1, 0, 0, 3] = np.exp(epsilon * 1j) * ctest[1, 0, 0, 3]
    ctest[1, 0, 1, 3] = np.exp(epsilon * 1j) * ctest[1, 0, 1, 3]
    ctest[1, 1, 0, 3] = np.exp(-epsilon * 1j) * ctest[1, 1, 0, 3]
    ctest[1, 1, 1, 3] = np.exp(-epsilon * 1j) * ctest[1, 1, 1, 3]
    '''dirac_operator = dirac.DiracWilsonOperator(
    lt.Configuration(gauge_field = tf.convert_to_tensor(ctest),
    geometry = geometry,
    colors = args.colors,
    number_of_dimensions = args.number_of_dimensions),
    kappa,
    hermitian=True, 
    representation = "fundamental")

    square_root_polynomial_approximation = dirac.PolynomialApproximation(
    dirac.SquareOperator(dirac_operator), 
    roots = square_root_polynomial_approximation_coefficients[1:], 
    scaling = square_root_polynomial_approximation_coefficients[0])

    overlap = dirac.Overlap(dirac_operator, square_root_polynomial_approximation, 0.2, hermitian = True)

    force_rational_approximation = dirac.RationalApproximation(
        dirac.SquareOperator(overlap),
        [1.0],
        [0.0],
        shift = 0,
        solver = multishift_solver)

    faction = fermion_action.n_flavor(
        [force_rational_approximation], 
        [force_rational_approximation], 
        [force_rational_approximation],
        multishift_solver,
        overlap)


    faction.pseudofermions = faction_original.pseudofermions

    S1 = faction.energy()'''

    faction_original.set_gauge_configuration(lt.Configuration(gauge_field=tf.convert_to_tensor(ctest),
                                                              geometry=geometry,
                                                              colors=colors,
                                                              number_of_dimensions=number_of_dimensions))

    S1 = faction_original.energy()

    ctest = config.gauge_field.numpy()
    ctest[1, 0, 0, 3] = np.exp(-epsilon * 1j) * ctest[1, 0, 0, 3]
    ctest[1, 0, 1, 3] = np.exp(-epsilon * 1j) * ctest[1, 0, 1, 3]
    ctest[1, 1, 0, 3] = np.exp(epsilon * 1j) * ctest[1, 1, 0, 3]
    ctest[1, 1, 1, 3] = np.exp(epsilon * 1j) * ctest[1, 1, 1, 3]
    '''dirac_operator = dirac.DiracWilsonOperator(
        lt.Configuration(gauge_field = tf.convert_to_tensor(ctest),
        geometry = geometry,
        colors = args.colors,
        number_of_dimensions = args.number_of_dimensions),
        kappa,
        hermitian=True, 
        representation = "fundamental")

    square_root_polynomial_approximation = dirac.PolynomialApproximation(
        dirac.SquareOperator(dirac_operator), 
        roots = square_root_polynomial_approximation_coefficients[1:], 
        scaling = square_root_polynomial_approximation_coefficients[0])

    overlap = dirac.Overlap(dirac_operator, square_root_polynomial_approximation, 0.2, hermitian = True)

    force_rational_approximation = dirac.RationalApproximation(
    dirac.SquareOperator(overlap),
    [1.0],
    [0.0],
    shift = 0,
    solver = multishift_solver)

    faction = fermion_action.n_flavor(
        [force_rational_approximation], 
        [force_rational_approximation], 
        [force_rational_approximation],
        multishift_solver,
        overlap)


    faction.pseudofermions = faction_original.pseudofermions

    S2 = faction.energy()'''

    faction_original.set_gauge_configuration(lt.Configuration(gauge_field=tf.convert_to_tensor(ctest),
                                                              geometry=geometry,
                                                              colors=colors,
                                                              number_of_dimensions=number_of_dimensions))

    S2 = faction_original.energy()

    print(der[1, :, :, 3])
    print((S1 - S2) / (8 * epsilon), "versus", der[1, 0, 0, 3])
    # der2 = faction.force()
    # print((S1-S2)/(8*epsilon), "versus", der2[1,0,0,3])
