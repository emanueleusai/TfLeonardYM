from ast import operator
import tensorflow as tf
import lattice as lt
import numpy as np
from scipy.linalg import lu
import lie_generators
import algebra_utils


class Operator:
    def __init__(self, hermitian=False):
        self.hermitian = hermitian


class ShiftedOperator(Operator):
    def __init__(self, operator_to_shift: Operator, shift: float):
        super().__init__(operator_to_shift.hermitian)
        self.operator_to_shift = operator_to_shift
        self.shift = shift

    def set_gauge_configuration(self, cfg):
        self.operator_to_shift.set_gauge_configuration(cfg)

    def apply(self, vector):
        return self.operator_to_shift.apply(vector) + self.shift * vector


class SquareOperator(Operator):
    def __init__(self, operator_to_square: Operator):
        super().__init__(operator_to_square.hermitian)
        self.operator_to_square = operator_to_square

    def set_gauge_configuration(self, cfg):
        self.operator_to_square.set_gauge_configuration(cfg)

    def apply(self, vector):
        return self.operator_to_square.apply(self.operator_to_square.apply(vector))


class RationalApproximation(Operator):
    def __init__(self, operator, alphas, betas, shift, solver):
        super().__init__(operator.hermitian)
        self.operator = operator
        self.alphas = alphas
        self.betas = betas
        self.shift = shift
        self.solver = solver

    def set_gauge_configuration(self, cfg):
        self.operator.set_gauge_configuration(cfg)

    def apply(self, vector):
        solutions = self.solver.solve(self.operator, vector, self.betas)

        result = self.shift * (vector)
        for solution, alpha in zip(solutions, self.alphas):
            result = result + alpha * solution

        return result

    def evaluate(self, x: float):
        result = self.shift

        for alpha, beta in zip(self.alphas, self.betas):
            result += alpha / (x + beta)

        return result

    def test_inverter(self, vector):
        solutions = self.solver.solve(self.operator, vector, self.betas)
        for solution in solutions:
            v = self.operator.apply(solution)
            print("It must be zero", algebra_utils.dot(v - vector, v - vector))


class PolynomialApproximation(Operator):
    def __init__(self, operator, roots, scaling):
        super().__init__(operator.hermitian)
        self.roots = roots
        self.scaling = scaling
        self.operator = operator

    def apply(self, vector):
        output = tf.identity(vector)
        for root in self.roots:
            output = self.scaling * (self.operator.apply(output) - root * output)

        return output

    def set_gauge_configuration(self, cfg):
        self.operator.set_gauge_configuration(cfg)

    def order(self):
        return len(self.roots)


class Overlap(Operator):
    def __init__(self, dirac_operator, square_root_approximation, mass, hermitian=False):
        super().__init__(hermitian)
        self.square_root_approximation = square_root_approximation
        self.dirac_operator = dirac_operator
        self.mass = mass

        self.number_of_dimensions = dirac_operator.number_of_dimensions
        self.number_of_colors = dirac_operator.number_of_colors
        self.volume = dirac_operator.volume
        self.spinor_dimension = self.dirac_operator.spinor_dimension
        self.representation = self.dirac_operator.representation

    def set_gauge_configuration(self, cfg):
        self.square_root_approximation.set_gauge_configuration(cfg)
        self.dirac_operator.set_gauge_configuration(cfg)

        self.number_of_dimensions = self.dirac_operator.number_of_dimensions
        self.number_of_colors = self.dirac_operator.number_of_colors
        self.volume = self.dirac_operator.volume
        self.spinor_dimension = self.dirac_operator.spinor_dimension

    def apply(self, vector):
        spinor_dimensions = vector.shape[0]
        sqrt_vector = self.square_root_approximation.apply(vector)
        output = self.dirac_operator.apply(sqrt_vector)
        if self.hermitian:
            result = []

            for nu in range(spinor_dimensions // 2):
                result.append(((1 + self.mass) / 2) * vector[nu] + ((1 - self.mass) / 2) * output[nu])
            for nu in range(spinor_dimensions // 2, spinor_dimensions):
                result.append((-(1 + self.mass) / 2) * vector[nu] + ((1 - self.mass) / 2) * output[nu])

            return tf.stack(result)
        else:
            result = []

            for nu in range(spinor_dimensions // 2):
                result.append(((1 + self.mass) / 2) * vector[nu] + ((1 - self.mass) / 2) * output[nu])
            for nu in range(spinor_dimensions // 2, spinor_dimensions):
                result.append(((1 + self.mass) / 2) * vector[nu] + (-(1 - self.mass) / 2) * output[nu])

            return tf.stack(result)

    def lie_derivative(self, X, Y):
        factor = (0.5 - self.mass / 2.)
        left_dirac_vectors = [tf.zeros_like(X) for _ in range(self.square_root_approximation.order() + 1)]
        right_dirac_vectors = [tf.zeros_like(X) for _ in range(self.square_root_approximation.order() + 1)]

        right_dirac_vectors = [Y]
        for root in reversed(self.square_root_approximation.roots):
            right_dirac_vectors.append(self.square_root_approximation.scaling * (
                    self.square_root_approximation.operator.apply(right_dirac_vectors[-1]) - root *
                    right_dirac_vectors[-1]))

        right_dirac_vectors = list(reversed(right_dirac_vectors))

        left_dirac_vectors = [self.dirac_operator.apply(X)]
        for root in self.square_root_approximation.roots:
            left_dirac_vectors.append(self.square_root_approximation.scaling * (
                    self.square_root_approximation.operator.apply(left_dirac_vectors[-1]) - tf.math.conj(root) *
                    left_dirac_vectors[-1]))

        fermion_force = factor * self.dirac_operator.lie_derivative(X, right_dirac_vectors[0])

        for i, root in enumerate(self.square_root_approximation.roots):
            tmp = self.dirac_operator.apply(left_dirac_vectors[i])
            fermion_force += (self.square_root_approximation.scaling * factor) * self.dirac_operator.lie_derivative(tmp,
                                                                                                                    right_dirac_vectors[
                                                                                                                        i + 1])

            tmp = self.dirac_operator.apply(right_dirac_vectors[i + 1])
            fermion_force += (self.square_root_approximation.scaling * factor) * self.dirac_operator.lie_derivative(
                left_dirac_vectors[i], tmp)

        return fermion_force

        for i in range(self.square_root_approximation.order()):
            XX = self.dirac_operator.apply(X)
            for j in range(0, i):
                root = self.square_root_approximation.roots[j]
                XX = self.square_root_approximation.scaling * (
                        self.square_root_approximation.operator.apply(XX) - tf.math.conj(root) * XX)

            print("TEST TEST", (XX - left_dirac_vectors[i])[0, :, 2])

            YY = Y
            for j in range(i + 1, self.square_root_approximation.order()):
                root = self.square_root_approximation.roots[j]
                YY = self.square_root_approximation.scaling * (
                        self.square_root_approximation.operator.apply(YY) - root * YY)

            print("TEST TEST TEST", (YY - right_dirac_vectors[i + 1])[0, :, 2])

            tmp = self.dirac_operator.apply(XX)
            fermion_force -= (self.square_root_approximation.scaling * factor) * self.dirac_operator.lie_derivative(tmp,
                                                                                                                    YY)

            tmp = self.dirac_operator.apply(YY)
            fermion_force -= (self.square_root_approximation.scaling * factor) * self.dirac_operator.lie_derivative(XX,
                                                                                                                    tmp)

        return fermion_force


@tf.function
def _multiply_in_adjoint_representation_2D_mth(input_vector, plus_projectors, minus_projectors, back_plus_projectors,
                                               back_minus_projectors, lookup_tables, gauge_field, cfg_down):
    print("Tracing")
    output_vector = tf.stack([input_vector[0], -input_vector[1]])

    # for mu in range(2):
    # U_mu(x) * (1 - gamma_mu) * psi(x+mu)
    half_spinor_plus = tf.einsum("mn,njkc->mjkc", plus_projectors[0, :1],
                                 input_vector)
    sup_vector = tf.gather(half_spinor_plus, lookup_tables.sup[0], axis=3)

    hopping_sup_vector = tf.einsum("ijc,mjkc,lkc->milc", gauge_field[0], sup_vector,
                                   tf.math.conj(gauge_field[0]), optimize="optimal")

    half_spinor_minus = tf.einsum("mn,njkc->mjkc", minus_projectors[0, :1],
                                  input_vector)
    sdn_vector = tf.gather(half_spinor_minus, lookup_tables.sdn[0], axis=3)

    hopping_sdn_vector = tf.einsum("jic,mjkc,klc->milc", tf.math.conj(cfg_down[0]), sdn_vector,
                                   cfg_down[0], optimize="optimal")

    # Reconstruct the full spinor using the
    # linear combinations given by the LU decomposition
    hoppings_0p = tf.einsum("mn,nijc->mijc", 2 * back_plus_projectors[0], hopping_sup_vector)
    hoppings_0m = tf.einsum("mn,nijc->mijc", 2 * back_minus_projectors[0], hopping_sdn_vector)

    half_spinor_plus = tf.einsum("mn,njkc->mjkc", plus_projectors[1, :1],
                                 input_vector)
    sup_vector = tf.gather(half_spinor_plus, lookup_tables.sup[1], axis=3)

    hopping_sup_vector = tf.einsum("ijc,mjkc,lkc->milc", gauge_field[1], sup_vector,
                                   tf.math.conj(gauge_field[1]), optimize="optimal")

    half_spinor_minus = tf.einsum("mn,njkc->mjkc", minus_projectors[1, :1],
                                  input_vector)
    sdn_vector = tf.gather(half_spinor_minus, lookup_tables.sdn[1], axis=3)

    hopping_sdn_vector = tf.einsum("jic,mjkc,klc->milc", tf.math.conj(cfg_down[1]), sdn_vector,
                                   cfg_down[1], optimize="optimal")

    # Reconstruct the full spinor using the
    # linear combinations given by the LU decomposition
    hoppings_1p = tf.einsum("mn,nijc->mijc", 2 * back_plus_projectors[1], hopping_sup_vector)
    hoppings_1m = tf.einsum("mn,nijc->mijc", 2 * back_minus_projectors[1], hopping_sdn_vector)

    # return output_vector
    return tf.math.add_n([hoppings_0p, hoppings_0m, hoppings_1p, hoppings_1m, output_vector])


class DiracWilsonOperator(Operator):
    def __init__(self,
                 cfg: lt.Configuration,
                 kappa,
                 hermitian=False,
                 mode="projected",
                 representation="adjoint",
                 boundary_condition_provider=None):
        super().__init__(hermitian)
        self.kappa = kappa

        self.mode = mode
        self.representation = representation

        if mode not in ("projected", "simple"):
            raise RuntimeError(f"Unknown dirac multiplication mode {mode}")

        if (mode, representation) == ("projected", "adjoint"):
            self.apply = self.multiply_in_adjoint_representation
        if (mode, representation) == ("simple", "adjoint"):
            self.apply = self.simple_multiply_in_adjoint_representation
        if (mode, representation) == ("projected", "fundamental"):
            self.apply = self.multiply_in_fundamental_representation
        if (mode, representation) == ("simple", "fundamental"):
            self.apply = self.simple_multiply_in_fundamental_representation

        self.boundary_condition_provider = boundary_condition_provider
        self.set_gauge_configuration(cfg)
        self.set_gamma_matrices()

    def set_gamma_matrices(self):
        if self.number_of_dimensions == 2:
            self.gamma = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]]], dtype=np.complex128)
            self.identity = np.array([[1, 0], [0, 1]], dtype=np.complex128)
            self.gammad = np.array([[1, 0], [0, -1]], dtype=np.complex128)
            self.gammad_diagonal = np.array([1, -1], dtype=np.complex128)
            self.spinor_dimension = 2
        elif self.number_of_dimensions == 4:
            self.gamma = np.array([[[0, 0, 0, -1j], [0, 0, -1j, 0], [0, 1j, 0, 0], [1j, 0, 0, 0]],
                                   [[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]],
                                   [[0, 0, -1j, 0], [0, 0, 0, 1j], [1j, 0, 0, 0], [0, -1j, 0, 0]],
                                   [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]]], dtype=np.complex128)
            self.identity = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.complex128)
            self.gammad = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=np.complex128)
            self.gammad_diagonal = np.array([1, 1, -1, -1], dtype=np.complex128)
            self.spinor_dimension = 4
        else:
            raise RuntimeError(f"Number of dimensions {self.number_of_dimensions} not supported for fermions!")

        self.plus_projectors = []
        self.minus_projectors = []
        self.back_plus_projectors = []
        self.back_minus_projectors = []

        for mu in range(self.cfg.number_of_dimensions):
            plus_projector = (self.identity - self.gamma[mu])
            if self.hermitian:
                plus_projector = self.gammad @ plus_projector

            self.plus_projectors.append(plus_projector)

            minus_projector = (self.identity + self.gamma[mu])
            if self.hermitian:
                minus_projector = self.gammad @ minus_projector

            self.minus_projectors.append(minus_projector)

            # Reconstruct the full spinor using the 
            # linear combinations given by the LU decomposition
            plp, _ = lu(plus_projector, permute_l=True)
            plm, _ = lu(minus_projector, permute_l=True)

            # Project back to full spinor
            back_projector_tensor_plus = np.zeros(shape=(self.spinor_dimension, self.spinor_dimension // 2),
                                                  dtype=np.complex128)
            for nu1 in range(0, self.spinor_dimension // 2):
                back_projector_tensor_plus[nu1, nu1] = self.kappa
            for nu1 in range(self.spinor_dimension // 2, self.spinor_dimension):
                for nu2 in range(self.spinor_dimension // 2):
                    back_projector_tensor_plus[nu1, nu2] = self.kappa * plp[nu1, nu2]

            self.back_plus_projectors.append(back_projector_tensor_plus)

            back_projector_tensor_minus = np.zeros(shape=(self.spinor_dimension, self.spinor_dimension // 2),
                                                   dtype=np.complex128)
            for nu1 in range(0, self.spinor_dimension // 2):
                back_projector_tensor_minus[nu1, nu1] = self.kappa
            for nu1 in range(self.spinor_dimension // 2, self.spinor_dimension):
                for nu2 in range(self.spinor_dimension // 2):
                    back_projector_tensor_minus[nu1, nu2] = self.kappa * plm[nu1, nu2]

            self.back_minus_projectors.append(back_projector_tensor_minus)

        self.plus_projectors = np.array(self.plus_projectors)
        self.minus_projectors = np.array(self.minus_projectors)
        self.back_plus_projectors = np.array(self.back_plus_projectors)
        self.back_minus_projectors = np.array(self.back_minus_projectors)

    def set_gauge_configuration(self, cfg):
        self.number_of_dimensions = cfg.gauge_field.shape[0]
        self.number_of_colors = cfg.gauge_field.shape[1]
        self.volume = cfg.gauge_field.shape[-1]
        self.cfg = cfg
        self.original_cfg = cfg

        if self.representation == "adjoint":
            self.fermion_gauge_field = 2. * tf.einsum(
                "ijk,mklc,nlo,mjoc->minc",
                lie_generators.generators(self.number_of_colors, "fundamental"),
                self.cfg.gauge_field,
                lie_generators.generators(self.number_of_colors, "fundamental"),
                tf.math.conj(self.cfg.gauge_field))
        else:
            self.fermion_gauge_field = self.cfg.gauge_field

        if self.boundary_condition_provider is not None:
            self.fermion_gauge_field = self.boundary_condition_provider(self.fermion_gauge_field)
            self.cfg = self.boundary_condition_provider(self.cfg)

        self.cfg_down = []
        self.original_cfg_down = []
        self.fermion_gauge_field_down = []
        for mu in range(cfg.number_of_dimensions):
            self.cfg_down.append(lt.translate(self.cfg.gauge_field[mu], mu, -1))
            self.original_cfg_down.append(lt.translate(self.original_cfg.gauge_field[mu], mu, -1))
            self.fermion_gauge_field_down.append(lt.translate(self.fermion_gauge_field[mu], mu, -1))

    def simple_multiply_in_adjoint_representation(self, input_vector):
        output_vector = tf.identity(input_vector)

        for mu in range(self.cfg.number_of_dimensions):
            # U_mu(x) * (1 - gamma_mu) * psi(x+mu)
            plus_projector = (self.identity - self.gamma[mu])
            plus_projected = tf.einsum("mn,njkc->mjkc", plus_projector, input_vector)
            sup_vector = lt.translate(plus_projected, mu, +1)

            hopping_sup_vector = tf.einsum("ijc,mjkc,lkc->milc", self.cfg.gauge_field[mu], sup_vector,
                                           tf.math.conj(self.original_cfg.gauge_field[mu]))

            # U_mu(x-mu)^\dag * (1 + gamma_mu) * psi(x-mu)
            minus_projector = (self.identity + self.gamma[mu])
            minus_projected = tf.einsum("mn,njkc->mjkc", minus_projector, input_vector)
            sdn_vector = lt.translate(minus_projected, mu, -1)

            hopping_sdn_vector = tf.einsum("jic,mjkc,klc->milc", tf.math.conj(self.cfg_down[mu]), sdn_vector,
                                           self.original_cfg_down[mu])

            output_vector += self.kappa * (hopping_sup_vector + hopping_sdn_vector)

        return self.apply_hermiticity(output_vector)

    def multiply_in_adjoint_representation(self, input_vector):
        output_vector = self.apply_hermiticity(input_vector)
        spinor_dimensions = input_vector.shape[0]

        hoppings = [output_vector]

        for mu in range(self.cfg.number_of_dimensions):
            # U_mu(x) * (1 - gamma_mu) * psi(x+mu)
            half_spinor_plus = tf.einsum("mn,njkc->mjkc", self.plus_projectors[mu][:spinor_dimensions // 2],
                                         input_vector)
            sup_vector = lt.translate(half_spinor_plus, mu, +1)

            hopping_sup_vector = tf.einsum("ijc,mjkc,lkc->milc", self.cfg.gauge_field[mu], sup_vector,
                                           tf.math.conj(self.original_cfg.gauge_field[mu]), optimize="optimal")

            # U_mu(x-mu)^\dag * (1 + gamma_mu) * psi(x-mu)
            # cfg_down = translate(self.cfg.gauge_field[mu], mu, -1)
            half_spinor_minus = tf.einsum("mn,njkc->mjkc", self.minus_projectors[mu][:spinor_dimensions // 2],
                                          input_vector)
            sdn_vector = lt.translate(half_spinor_minus, mu, -1)

            hopping_sdn_vector = tf.einsum("jic,mjkc,klc->milc", tf.math.conj(self.cfg_down[mu]), sdn_vector,
                                           self.original_cfg_down[mu], optimize="optimal")

            '''hopping = [tf.zeros(input_vector.shape[1:], tf.complex128) for _ in range(self.spinor_dimension)]

            for nu in range(spinor_dimensions//2):
                hopping[nu] += (2*self.kappa)*(hopping_sup_vector[nu] + hopping_sdn_vector[nu])
        
            # Reconstruct the full spinor using the 
            # linear combinations given by the LU decomposition
            plp, _ = lu(plus_projector, permute_l=True)
            plm, _ = lu(minus_projector, permute_l=True)

            for nu1 in range(spinor_dimensions//2,spinor_dimensions):
                for nu2 in range(spinor_dimensions//2):
                    if plp[nu1,nu2] != 0 and plm[nu1,nu2] != 0:
                        hopping[nu1] += (2*self.kappa)*(plp[nu1,nu2]*hopping_sup_vector[nu2] + plm[nu1,nu2]*hopping_sdn_vector[nu2])
                    elif plp[nu1,nu2] != 0:
                        hopping[nu1] += (2*self.kappa*plp[nu1,nu2])*hopping_sup_vector[nu2]
                    elif plm[nu1,nu2] != 0:
                        hopping[nu1] += (2*self.kappa*plm[nu1,nu2])*hopping_sdn_vector[nu2]
        
            # Finally add the hopping term
            output_vector += tf.stack(hopping)'''

            # Reconstruct the full spinor using the 
            # linear combinations given by the LU decomposition
            hoppings.append(tf.einsum("mn,nijc->mijc", self.back_plus_projectors[mu], hopping_sup_vector))
            hoppings.append(tf.einsum("mn,nijc->mijc", self.back_minus_projectors[mu], hopping_sdn_vector))

        # return output_vector
        return tf.math.add_n(hoppings)

    def multiply_in_fundamental_representation(self, input_vector):
        output_vector = tf.identity(input_vector)
        spinor_dimensions = input_vector.shape[0]

        for mu in range(self.cfg.number_of_dimensions):
            # U_mu(x) * (1 - gamma_mu) * psi(x+mu)
            plus_projector = (self.identity - self.gamma[mu])
            half_spinor_plus = tf.einsum("mn,njc->mjc", plus_projector[:spinor_dimensions // 2], input_vector)
            sup_vector = lt.translate(half_spinor_plus, mu, +1)

            hopping_sup_vector = tf.einsum("ijc,mjc->mic", self.fermion_gauge_field[mu], sup_vector)

            # U_mu(x-mu)^\dag * (1 + gamma_mu) * psi(x-mu)
            # cfg_down = translate(self.cfg.gauge_field[mu], mu, -1)
            minus_projector = (self.identity + self.gamma[mu])
            half_spinor_minus = tf.einsum("mn,njc->mjc", minus_projector[:spinor_dimensions // 2], input_vector)
            sdn_vector = lt.translate(half_spinor_minus, mu, -1)

            hopping_sdn_vector = tf.einsum("jic,mjc->mic", tf.math.conj(self.fermion_gauge_field_down[mu]), sdn_vector)

            hopping = [tf.zeros(input_vector.shape[1:], tf.complex128) for _ in range(self.spinor_dimension)]

            for nu in range(spinor_dimensions // 2):
                hopping[nu] += self.kappa * (hopping_sup_vector[nu] + hopping_sdn_vector[nu])

            # Reconstruct the full spinor using the 
            # linear combinations given by the LU decomposition
            plp, _ = lu(plus_projector, permute_l=True)
            plm, _ = lu(minus_projector, permute_l=True)

            for nu1 in range(spinor_dimensions // 2, spinor_dimensions):
                for nu2 in range(spinor_dimensions // 2):
                    if plp[nu1, nu2] != 0 and plm[nu1, nu2] != 0:
                        hopping[nu1] += self.kappa * (
                                plp[nu1, nu2] * hopping_sup_vector[nu2] + plm[nu1, nu2] * hopping_sdn_vector[nu2])
                    elif plp[nu1, nu2] != 0:
                        hopping[nu1] += (self.kappa * plp[nu1, nu2]) * hopping_sup_vector[nu2]
                    elif plm[nu1, nu2] != 0:
                        hopping[nu1] += (self.kappa * plm[nu1, nu2]) * hopping_sdn_vector[nu2]

            # Finally add the hopping term
            output_vector += tf.stack(hopping)

        return self.apply_hermiticity(output_vector)

    def simple_multiply_in_fundamental_representation(self, input_vector):
        output_vector = tf.identity(input_vector)

        for mu in range(self.cfg.number_of_dimensions):
            # U_mu(x) * (1 - gamma_mu) * psi(x+mu)
            plus_projector = (self.identity - self.gamma[mu])
            plus_projected = tf.einsum("mn,njc->mjc", plus_projector, input_vector)
            sup_vector = lt.translate(plus_projected, mu, +1)

            hopping_sup_vector = tf.einsum("ijc,mjc->mic", self.cfg.gauge_field[mu], sup_vector)

            # U_mu(x-mu)^\dag * (1 + gamma_mu) * psi(x-mu)
            minus_projector = (self.identity + self.gamma[mu])
            minus_projected = tf.einsum("mn,njc->mjc", minus_projector, input_vector)
            sdn_vector = lt.translate(minus_projected, mu, -1)

            hopping_sdn_vector = tf.einsum("jic,mjc->mic", tf.math.conj(self.cfg_down[mu]), sdn_vector)

            output_vector += self.kappa * (hopping_sup_vector + hopping_sdn_vector)

        return self.apply_hermiticity(output_vector)

    def lie_derivative(self, X, Y):
        derivatives = []

        if len(X.shape) == 4:
            # Adjoint representation
            # Convert the adjoint matrix to an adjoint vector
            X = 2 * tf.einsum("ijk,mkjc->mic", lie_generators.generators(self.number_of_colors, "fundamental"), X)
            Y = 2 * tf.einsum("ijk,mkjc->mic", lie_generators.generators(self.number_of_colors, "fundamental"), Y)

        for mu in range(self.number_of_dimensions):
            plus_projector = (self.identity - self.gamma[mu])
            if self.hermitian:
                plus_projector = self.gammad @ plus_projector

            minus_projector = (self.identity + self.gamma[mu])
            if self.hermitian:
                minus_projector = self.gammad @ minus_projector

            Y_sup = lt.translate(Y, mu, +1)

            # Derivative from X(x)^\dag\gamma_5(1-\gamma_\mu) U_\mu(x)Y(x+mu)
            derivative = tf.einsum("mis,mn,kij,jls,nls,kut->uts",
                                   tf.math.conj(X),
                                   plus_projector,
                                   lie_generators.generators(self.number_of_colors, self.representation),
                                   self.fermion_gauge_field[mu],
                                   Y_sup,
                                   lie_generators.generators(self.number_of_colors, "fundamental"))

            # Derivative from Y(x+mu)^\dag\gamma_5(1+\gamma_\mu) U^\dag_\mu(x)X(x)
            derivative -= tf.einsum("mis,mn,jis,kjl,nls,kut->uts",
                                    tf.math.conj(Y_sup),
                                    minus_projector,
                                    tf.math.conj(self.fermion_gauge_field[mu]),
                                    lie_generators.generators(self.number_of_colors, self.representation),
                                    X,
                                    lie_generators.generators(self.number_of_colors, "fundamental"))

            derivatives.append(derivative)

        return self.kappa * tf.stack(derivatives)

    def test(self, X, Y):
        derivatives = []

        if len(X.shape) == 4:
            # Adjoint representation
            # Convert the adjoint matrix to an adjoint vector
            X = 2. * tf.einsum("ijk,mkjc->mic", lie_generators.generators(self.number_of_colors, "fundamental"), X)
            Y = 2. * tf.einsum("ijk,mkjc->mic", lie_generators.generators(self.number_of_colors, "fundamental"), Y)

        for mu in range(self.number_of_dimensions):
            plus_projector = (self.identity - self.gamma[mu])
            if self.hermitian:
                plus_projector = self.gammad @ plus_projector

            Y_sup = lt.translate(Y, mu, +1)
            X_sup = lt.translate(X, mu, +1)

            # Derivative from X(x)^\dag\gamma_5(1-\gamma_\mu) U_\mu(x)Y(x+mu)
            derivative = ((1j * self.kappa) *
                          tf.einsum("ic,ijk->jkc",
                                    tf.cast(
                                        (tf.einsum("mic,mn,kij,jlc,nlc->kc",
                                                   tf.math.conj(X),
                                                   plus_projector,
                                                   lie_generators.generators(self.number_of_colors,
                                                                             self.representation),
                                                   self.fermion_gauge_field[mu],
                                                   Y_sup)),
                                        dtype=tf.complex128),
                                    lie_generators.generators(self.number_of_colors, "fundamental"),
                                    ))

            # Derivative from Y(x)^\dag\gamma_5(1-\gamma_\mu) U_\mu(x)X(x+mu)
            derivative += ((1j * self.kappa) *
                           tf.einsum("ic,ijk->jkc",
                                     tf.cast(
                                         (tf.einsum("mic,mn,kij,jlc,nlc->kc",
                                                    tf.math.conj(Y),
                                                    plus_projector,
                                                    lie_generators.generators(self.number_of_colors,
                                                                              self.representation),
                                                    self.fermion_gauge_field[mu],
                                                    X_sup)),
                                         dtype=tf.complex128),
                                     lie_generators.generators(self.number_of_colors, "fundamental"))
                           )

            minus_projector = (self.identity + self.gamma[mu])
            if self.hermitian:
                minus_projector = self.gammad @ minus_projector

            # Derivative from X(x+mu)^\dag\gamma_5(1+\gamma_\mu) U^\dag_\mu(x)Y(x)
            derivative -= ((1j * self.kappa) *
                           tf.einsum("ic,ijk->jkc",
                                     tf.cast(
                                         (tf.einsum("mic,mn,jic,kja,nac->kc",
                                                    tf.math.conj(X_sup),
                                                    minus_projector,
                                                    tf.math.conj(self.fermion_gauge_field[mu]),
                                                    lie_generators.generators(self.number_of_colors,
                                                                              self.representation),
                                                    Y)), dtype=tf.complex128),
                                     lie_generators.generators(self.number_of_colors, "fundamental"),
                                     ))

            # Derivative from Y(x+mu)^\dag\gamma_5(1+\gamma_\mu) U^\dag_\mu(x)X(x)
            derivative -= ((1j * self.kappa) *
                           tf.einsum(
                               "ic,ijk->jkc",
                               tf.cast(
                                   (tf.einsum("mic,mn,jic,kja,nac->kc",
                                              tf.math.conj(Y_sup),
                                              minus_projector,
                                              tf.math.conj(self.fermion_gauge_field[mu]),
                                              lie_generators.generators(self.number_of_colors, self.representation),
                                              X)), dtype=tf.complex128),
                               lie_generators.generators(self.number_of_colors, "fundamental")))

            derivatives.append(derivative)

        return tf.stack(derivatives)

    def apply_hermiticity(self, input_vector):
        spinor_dimensions = input_vector.shape[0]

        if not self.hermitian:
            return tf.identity(input_vector)

        result = []

        for nu in range(spinor_dimensions // 2):
            result.append(input_vector[nu])
        for nu in range(spinor_dimensions // 2, spinor_dimensions):
            result.append(-input_vector[nu])

        return tf.stack(result)


class Propagator:
    def __init__(self, operator, inverter):
        self.operator = operator
        self.inverter = inverter

    def apply(self, source):
        square_solution = self.inverter.solve(SquareOperator(self.operator), source)
        solution = self.operator.apply(square_solution)

        if not isinstance(self.operator, Overlap):
            return self.apply_gammad(solution)

        mass = self.operator.mass
        self.operator.mass = 0

        result = solution - self.operator.apply(solution)

        self.operator.mass = mass

        return self.apply_gammad(result)

    def apply_gammad(self, input_vector):
        spinor_dimensions = input_vector.shape[0]

        result = []

        for nu in range(spinor_dimensions // 2):
            result.append(input_vector[nu])
        for nu in range(spinor_dimensions // 2, spinor_dimensions):
            result.append(-input_vector[nu])

        return tf.stack(result)
