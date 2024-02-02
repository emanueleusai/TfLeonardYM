import tensorflow as tf
import numpy as np
import algebra_utils
import dirac


class n_flavor:
    def __init__(self,
                 force_operators: dirac.RationalApproximation,
                 metropolis_operators: dirac.RationalApproximation,
                 heatbath_operators: dirac.RationalApproximation,
                 solver,
                 dirac_operator: dirac.Operator):
        self.force_operators = force_operators
        self.metropolis_operators = metropolis_operators
        self.heatbath_operators = heatbath_operators
        self.solver = solver
        self.dirac_operator = dirac_operator
        self.square_dirac_operator = dirac.SquareOperator(dirac_operator)
        self.pseudofermions = None

    def initialize_pseudofermions(self, random_vectors):
        self.pseudofermions = [
            operator.apply(vector)
            for vector, operator in zip(random_vectors, self.heatbath_operators)]

    def set_gauge_configuration(self, cfg):
        self.dirac_operator.set_gauge_configuration(cfg)
        self.square_dirac_operator = dirac.SquareOperator(self.dirac_operator)
        for approximation in self.force_operators:
            approximation.set_gauge_configuration(cfg)

        for approximation in self.metropolis_operators:
            approximation.set_gauge_configuration(cfg)

        for approximation in self.heatbath_operators:
            approximation.set_gauge_configuration(cfg)

    def energy(self):
        energy = 0
        for vector, operator in zip(self.pseudofermions, self.metropolis_operators):
            energy += algebra_utils.dot(vector, operator.apply(vector))

        return tf.math.real(energy)

    def number_of_flavors(self):
        result = 1

        for approximation in self.metropolis_operators:
            result *= approximation.evaluate(0.5)

        return 2 * np.log(result) / np.log(2.)

    def check_rational_approximations(self, vector):
        result = 0.
        for metropolis, heatbath in zip(self.metropolis_operators, self.heatbath_operators):
            first = heatbath.apply(vector)
            second = metropolis.apply(first)
            third = heatbath.apply(second)
            result += algebra_utils.dot(third - vector, third - vector)

        return result

    def force(self):
        # First we solve the dirac equation for all shifts and all pseudofermions
        Xs = [
            self.solver.solve(force_operator.operator, pseudofermion, force_operator.betas)
            for force_operator, pseudofermion in zip(self.force_operators, self.pseudofermions)]

        # Then we multiply each solution by the dirac operator
        Ys = [[self.dirac_operator.apply(solution) for solution in x] for x in Xs]

        # Zs = [[self.dirac_operator.apply(solution) for solution in y] for y in Ys]

        # for Z, pseudofermion in zip(Zs, self.pseudofermions):
        #    for z in Z:
        #        print("Force test", algebra_utils.dot(z-pseudofermion, z- pseudofermion))

        force = tf.zeros(
            shape=(
                self.dirac_operator.number_of_dimensions,
                self.dirac_operator.number_of_colors,
                self.dirac_operator.number_of_colors,
                self.dirac_operator.volume),
            dtype=tf.complex128)

        # for each rational approximation
        for X, Y, approximation in zip(Xs, Ys, self.force_operators):
            # for each term of the rational approximation
            for x, y, weight in zip(X, Y, approximation.alphas):
                force += weight * (self.dirac_operator.lie_derivative(x, y) + self.dirac_operator.lie_derivative(y, x))

        return force
