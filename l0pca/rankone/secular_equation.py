"""Secular equation solver.

Matrix shape will include two batch dimensions. This is in case we have multiple
n-1 x n-1 matrices, but for each one, we want to broadcast that matrix by
inserting a second dim and considering several perturbations.

Initial estimates shape assumes that we need exactly one estimate in-range for
each perturbed eigenvalue. They will be lined up with the cumsum of equation
terms (fitting a psi and phi function).
"""
from l0pca.util import gather_util
import tensorflow as tf

def eq_search_eigenvalues_iteration(diag, update_vec, initial_estimates):
    psi_values, phi_values = eq_evaluate(diag, update_vec, initial_estimates)
    psi_constant, psi_multiplier = fit_interpolation(
        psi_values, diag, initial_estimates)
    rho = tf.square(tf.linalg.norm(update_vec, axis=-1))
    # The theoretical limit for the final eigenvalue is the quantity being
    # added to the matrix trace.
    phi_diagonal_values = tf.concat(
        [
            diag[..., 1:],
            # Upper bound where all perturbed eigenvalues go to 0, except for
            # this final one which goes to the trace of the matrix. The use of
            # fitting quadratic values is only to say that the pole of the
            # secular function is already given. We are not giving this as a
            # feasible closed interval for the eigenvalue.
            (tf.math.reduce_sum(diag, axis=-1) + rho)[:, :, None],
        ],
        axis=-1,
    )
    phi_constant, phi_multiplier = fit_interpolation(
        phi_values, phi_diagonal_values, initial_estimates)
    q_a, q_b, q_c = fit_quadratic(
        # The unit-norm update is pre-muliplied by rho. When the update vec
        # components already encode a norm not equal to 1, then the additional
        # constant term is 1, otherwise it could be 1/rho.
        1. + psi_constant + phi_constant,
        psi_multiplier,
        diag,
        phi_multiplier,
        phi_diagonal_values)
    det = tf.math.sqrt(q_b ** 2 - 4 * q_a * q_c)
    solution_one = (-q_b - det) / (2 * q_a)
    solution_two = (-q_b + det) / (2 * q_a)
    objective_one = tf.math.abs(solution_one - initial_estimates)
    objective_two = tf.math.abs(solution_two - initial_estimates)
    return tf.where(
        (diag < solution_one) & (solution_one < phi_diagonal_values)
        & (objective_one <= objective_two),
        solution_one,
        solution_two)

def eq_terms(diag, update_vec, initial_estimates):
    # 2 batch dims are expected.
    n = tf.shape(diag)[-1]
    diag = tf.ensure_shape(diag, [None, None, n])
    update_vec = tf.ensure_shape(update_vec, [None, None, n])
    initial_estimates = tf.ensure_shape(initial_estimates, [None, None, n])

    # Output shape:
    # 1: batch1
    # 2: batch2
    # 3: Power (secular function, or its first derivative)
    # 4: Current estimates of perturbed eigenvalues
    # 5: Secular equation term depending on each eigenvalue (to be summed up)
    numerator = tf.square(update_vec)[:, :, None, None, :]
    denominator = tf.math.pow(
        diag[:, :, None, None, :] - initial_estimates[:, :, None, :, None],
        tf.constant([-1, -2], numerator.dtype)[None, None, :, None, None])
    return numerator * denominator

def eq_evaluate(diag, update_vec, initial_estimates):
    terms = eq_terms(diag, update_vec, initial_estimates)
    # Psi function: Terms depend on eigenvalues which are less than the current
    # perturbed estimate. The smallest perturbed estimate should have one
    # eigenvalue smaller, and the largest perturbed estimate should be larger
    # than all eigenvalues (in theory, it is bounded by the trace of the new
    # matrix).
    psi_values = tf.cumsum(terms, axis=-1)
    # Which psi function are we interested in? They lie along a diagonal of Nth
    # current estimate (ordered) and Nth eigenvalue term (ordered).
    psi_values = tf.linalg.diag_part(psi_values)
    # psi_i(x) ~= psi_constant + psi_muliplier / (diag_i - x), for the ith
    # perturbed eigenvalue.
    psi_constant = (
        psi_values[:, :, 0, :]
        - psi_values[:, :, 1, :] * (diag - initial_estimates))
    psi_multiplier = (diag - initial_estimates) ** 2 * psi_values[:, :, 1, :]

    # The phi function (terms given by eigenvalues larger than the current
    # search interval) requires a reverse cumsum. The result is indexed off by
    # one (no eigenvalue here is smaller than every original eigenvalue). The
    # final phi function is a constant 0, so the exact eigenvalue would be
    # easier to solve.
    phi_values = tf.cumsum(terms, axis=-1, reverse=True)
    # Original: phi_values[0] summed all terms, phi_values[1] excluded first
    # term, ...
    phi_values = tf.concat(
        [
            phi_values[..., 1:],
            tf.zeros_like(phi_values[..., 0:1]),
        ],
        axis=-1,
    )
    phi_values = tf.linalg.diag_part(phi_values)
    return psi_values, phi_values

def fit_interpolation(values, diag, initial_estimates):
    constant_term = (
        values[:, :, 0, :]
        - values[:, :, 1, :] * (diag - initial_estimates)
    )
    multiplier_term = (diag - initial_estimates) ** 2 * values[:, :, 1, :]
    return constant_term, multiplier_term

def fit_quadratic(
    constant_term, multiplier_one, diag_one, multiplier_two, diag_two):
    # Equation: c + m1/(di - x) + m2/(d_{i+1} - x) = 0
    # We have:
    # c * (di - x) * (d_{i+1} - x) + m1 * (d_{i+1} - x) + m2 * (di - x) = 0
    # A x^2 + B x + C = 0, where:
    # A = c
    # B = -c (di + d_{i+1}) - m1 - m2
    # C = c di d_{i+1} + m1 d_{i+1} + m2 di
    q_a = constant_term
    q_b = -constant_term * (diag_one + diag_two) - multiplier_one - multiplier_two
    q_c = (
        constant_term * diag_one * diag_two
            + multiplier_one * diag_two + multiplier_two * diag_one)
    return q_a, q_b, q_c