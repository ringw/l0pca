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

def secular_equation_terms(diag, update_vec, initial_estimates):
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
    numerator = tf.square(diag)[:, :, None, None, :]
    denominator = tf.power(
        diag[:, :, None, None, :] - initial_estimates[:, :, None, :, None],
        tf.constant([-1, -2])[None, None, :, None, None])
    return numerator / denominator

def secular_equation_fit(diag, terms, initial_estimates):
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