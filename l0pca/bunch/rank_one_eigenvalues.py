from l0pca.bunch import pade
from l0pca.bunch import rank_one_update
import tensorflow as tf

def eigenvalue_search(update_vec, S):
    smallest_eig = tf.math.square(S[..., 0])
    search_smallest = tf.linspace(smallest_eig * 0.5, smallest_eig * 0.9, 3)
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
    eig_search_linspace = tf.linspace(
        start=tf.fill(eig_buckets[..., :-1].shape, tf.constant(0.1, S.dtype)),
        stop=0.9, num=3)
    eig_search = eig_buckets[..., None, :-1] + (eig_buckets[..., None, 1:] - eig_buckets[..., None, :-1]) * eig_search_linspace
    eig_search = tf.concat(
        [
            search_smallest[..., :, None],
            eig_search,
        ],
        axis=-1,
    )
    taylor = rank_one_update.bunch_rational_function_taylor_series(update_vec, S, eig_search)
    return 0.
