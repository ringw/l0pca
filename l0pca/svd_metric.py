import numpy as np
from l0pca import sampling
import tensorflow as tf

N_ENSEMBLE = 1024

@tf.function
def gather_cov(cov, batches):
    inds_row = batches[..., :, None, None]
    inds_col = batches[..., None, :, None]
    inds_shape = tf.broadcast_dynamic_shape(
        tf.shape(inds_row), tf.shape(inds_col))
    inds = tf.concat(
        [
            tf.broadcast_to(inds_row, inds_shape),
            tf.broadcast_to(inds_col, inds_shape),
        ],
        axis=-1)
    return tf.gather_nd(cov, inds)

def evaluate_cov_eigh(cov, weights, k, ndim, param_formula=None):
    n_features = cov.shape[0]
    n_ensemble_batch = tf.maximum(
        k,
        tf.math.floordiv(n_features, 10))
    n_ensemble_batch = tf.math.floordiv(n_ensemble_batch, k)
    n_ensemble_replicate = tf.math.floordiv(N_ENSEMBLE, n_ensemble_batch) + 1
    ensembles = sampling.weighted_choice_naive(weights, [n_ensemble_replicate, n_ensemble_batch * k])
    n_ensemble_actual = n_ensemble_replicate * n_ensemble_batch
    ensembles = tf.reshape(ensembles, [n_ensemble_actual, k])
    ensemble_cov = gather_cov(cov, ensembles)

    # n_data = tf.cast(tf.math.ceil(n_features * DATA_FRAC), tf.int32)
    # data_lookup = sampling.weighted_choice_naive(weights, [1, n_data])[0, :]
    n_data = cov.shape[0]
    data_lookup = tf.range(cov.shape[0])

    # Extract data rows of cov.
    data_rows = tf.gather(cov, data_lookup)
    # Extract ensemble columns of cov.
    ensembles_flat = tf.reshape(ensembles, [n_ensemble_actual * k])
    data_vectors = tf.transpose(
        tf.gather(
            tf.transpose(data_rows),
            ensembles_flat))
    data_vectors = tf.reshape(data_vectors, [n_data, n_ensemble_actual, k])
    data_diag = tf.gather_nd(
        cov,
        tf.tile(data_lookup[:, None], multiples=[1, 2]))

    # Loss metric (for graphing).
    ensemble_evalues, ensemble_evectors = tf.linalg.eigh(ensemble_cov)
    ensemble_evalues = tf.ensure_shape(
        ensemble_evalues,
        [None, k],
    )
    loss = -tf.math.reduce_mean(
        tf.linalg.norm(
            tf.sort(ensemble_evalues, axis=1)[:, -1:],
            axis=1))
    pc_component_stddev = tf.math.reduce_std(
        tf.math.reduce_min(tf.math.abs(ensemble_evectors[:, -1, :]), axis=1))
    # Now tile the ensemble matrices with one added data feature.
    ensemble_cov = tf.broadcast_to(
        ensemble_cov[None, :, :, :], [n_data, n_ensemble_actual, k, k])

    augmented_cov = tf.concat(
        [
            tf.concat(
                [
                    ensemble_cov,
                    data_vectors[:, :, :, None],
                ],
                axis=3,
            ),
            tf.concat(
                [
                    data_vectors[:, :, None, :],
                    tf.broadcast_to(
                        data_diag[:, None, None, None],
                        [n_data, n_ensemble_actual, 1, 1],
                    ),
                ],
                axis=3,
            )
        ],
        axis=2,
    )
    evalues, evectors = tf.linalg.eigh(augmented_cov)

    pc0 = tf.gather(
        tf.transpose(evectors, [0, 1, 3, 2]),
        tf.argmax(evalues, axis=2),
        batch_dims=2,
    )
    score = tf.math.abs(pc0[:, :, -1]) - tf.math.reduce_min(
        tf.math.abs(pc0[:, :, :-1]),
        axis=2)
    score_na_conflict = tf.reduce_any(
        data_lookup[:, None, None] == ensembles[None, :, :],
        axis=2)
    score = tf.where(score < 0., 0., score)
    score = tf.where(
        score_na_conflict, tf.constant(np.nan, tf.float32), score)
    def bernoulli(score):
        return tf.experimental.numpy.nanmean(tf.math.sign(score), axis=1)
    def exp_diff(score):
        return tf.experimental.numpy.nanmean(score, axis=1)
    def log_pulldown(score):
        score = tf.experimental.numpy.nanmean(score, axis=1)
        threshold = 1e-12
        return tf.where(score < threshold, 0., tf.math.log(score) / -tf.math.log(threshold))
    def tanh_activation(score):
        return tf.experimental.numpy.nanmean(
            tf.math.tanh(score / pc_component_stddev), axis=1)
    param_formula_fn = {
        None: exp_diff,
        'exp_diff': exp_diff,
        'bernoulli': bernoulli,
        'log_pulldown': log_pulldown,
        'tanh_activation': tanh_activation,
    }[param_formula]
    return loss, data_lookup, param_formula_fn(score)
