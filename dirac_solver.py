import tensorflow as tf
import numpy as np
import dirac
import algebra_utils


@tf.function
def _multiply_add(s, r, m):
    print("Tracing multiply add")
    return s + m * r


class multishift_solver:
    def __init__(self, max_steps, precision, omega=0.95):
        self.max_steps = max_steps
        self.precision = precision
        self.omega = omega

    def solve(self, operator: dirac.Operator, source, shifts: list):
        solutions = [tf.zeros_like(source) for _ in shifts]
        f = [1 for _ in shifts]
        r = tf.identity(source)

        initial_shift_index = 0

        while shifts[initial_shift_index] < 0:
            solver = biconjugate_gradient(self.max_steps, self.precision)
            shifted_operator = dirac.ShiftedOperator(operator, shifts[initial_shift_index])
            solutions[initial_shift_index] = solver.solve(shifted_operator, source)
            initial_shift_index += 1

        for step in range(self.max_steps):
            p = operator.apply(r) + shifts[-1] * r

            alpha = self.omega * algebra_utils.dot(p, r) / algebra_utils.dot(p, p)

            solutions[-1] = solutions[-1] + alpha * r

            for i, shift in enumerate(shifts[initial_shift_index:-1]):
                f[initial_shift_index + i] = f[initial_shift_index + i] / (1 + (shift - shifts[-1]) * alpha)

                solutions[initial_shift_index + i] = solutions[initial_shift_index + i] + alpha * f[
                    initial_shift_index + i] * r

            r = r - alpha * p

            error = tf.math.abs(algebra_utils.dot(r, r))
            if error < self.precision:
                print(f"MMMRMultishiftsolver: convergence after {step} steps")
                return solutions

        print(f"MMMRMultishiftsolver: failed to find convergence after {step} steps")
        return solutions


class biconjugate_gradient:
    def __init__(self, max_steps, precision):
        self.max_steps = max_steps
        self.precision = precision

    def solve(self, operator: dirac.Operator, source, initial_guess=None):
        if initial_guess is None:
            solution = tf.identity(source)
        else:
            solution = tf.identity(initial_guess)

        init = operator.apply(solution)

        residual = solution - init
        residual_hat = solution + init
        alpha = 1
        omega = 1
        rho = 1

        p = tf.zeros_like(source)
        nu = tf.zeros_like(source)

        for step in range(self.max_steps):
            rho_next = algebra_utils.dot(residual_hat, residual)

            beta = (rho_next / rho) * (alpha / omega)

            p = residual + beta * (p - omega * nu)

            nu = operator.apply(p)

            alpha = rho_next / algebra_utils.dot(residual_hat, nu)

            s = residual - alpha * nu

            t = operator.apply(s)

            omega = algebra_utils.dot(t, s) / algebra_utils.dot(t, t)

            solution = solution + alpha * p + omega * s

            residual = s - omega * t

            norm = tf.math.abs(algebra_utils.dot(residual, residual))

            if norm < self.precision:
                print(f"BiconjugateGradient: convergence after {step} steps")
                return solution

            residual = s - omega * t

            rho = rho_next

        print(f"BiconjugateGradient: failed to find convergence after {step} steps")
        return solution
