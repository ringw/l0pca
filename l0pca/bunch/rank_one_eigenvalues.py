from l0pca.util import batch_matrices
import numpy as np
from l0pca.bunch import pade
from l0pca.bunch import rank_one_update
import tensorflow as tf

#@tf.function(jit_compile=True)
def eigenvalue_search_seeds(update_vec, S):
    smallest_eig = tf.math.square(S[..., 0])
    search_smallest = tf.linspace(smallest_eig * 0.5, smallest_eig * 0.99, 15)
    # What are the intervals that bound each new eigenvalue? For the single
    # eigenvalue which is strictly larger than the previous largest eigenvalue,
    # it is bounded by the trace of the new matrix.
    eig_buckets = tf.concat(
        [
            tf.math.square(S),
            (tf.math.reduce_sum(tf.math.square(S), axis=-1)
                + tf.math.reduce_sum(tf.math.square(update_vec), axis=-1))[
                    ..., None],
        ],
        axis=-1,
    )
    # Lin space of eigenvalue initial guesses, inserted as zeroth dim.
    eig_search_linspace = tf.linspace(
        start=tf.fill(eig_buckets[..., :-1].shape, tf.constant(0.01, S.dtype)),
        stop=0.99, num=15)
    eig_search = eig_buckets[None, ..., :-1] + (eig_buckets[None, ..., 1:] - eig_buckets[None, ..., :-1]) * eig_search_linspace
    eig_search = tf.concat(
        [
            # For each eigenvalue, the search is along axis -2.
            search_smallest[..., :, None],
            eig_search,
        ],
        axis=-1,
    )
    # We expect not just the upper bounds on eigs, but a lower and upper bound
    # on the smallest eigenvalue (lower bound is 0).
    eig_buckets = tf.concat(
        [
            tf.zeros_like(eig_buckets[..., 0:1]),
            eig_buckets,
        ],
        axis=-1,
    )
    # Not desirable to have the search axis as the zeroth axis (it is being
    # aggregated away soon), so move it to -1.
    return eig_buckets, tf.experimental.numpy.moveaxis(eig_search, 0, -1)

#@tf.function(jit_compile=True)
def eigenvalue_initial_estimate(update_vec, S):
    eig_buckets, eig_search = eigenvalue_search_seeds(update_vec, S)
    taylor = rank_one_update.bunch_rational_function_taylor_series(
        update_vec, S, eig_search)
    eig_init = eig_search + pade.pade_zero_find(taylor)
    def squeeze_last(t):
        return t[..., 0]
    eig_goodness_of_fit = squeeze_last(
        tf.math.abs(
            rank_one_update.bunch_rational_function_taylor_series(
                update_vec, S, eig_init, num_order=1)))
    eig_goodness_of_fit = tf.where(
        (eig_buckets[..., :-1, None] < eig_init)
            & (eig_init < eig_buckets[..., 1:, None]),
        eig_goodness_of_fit,
        tf.fill(eig_goodness_of_fit.shape, tf.constant(np.inf, eig_buckets.dtype)))
    select_eig = tf.math.argmin(eig_goodness_of_fit, axis=-1)
    return tf.gather(
        eig_init,
        select_eig,
        batch_dims=len(eig_init.shape) - 1)