"""Main covariance matrix data object.

This is a singleton over the lifetime of the PCA problem. Number of variables
should be limited (1000) so that holding the Gram matrix here does not consume
excessive RAM. The system should generalize well to a full-rank, many more
observations, covariance matrix, so computing the Gram matrix on-device right
away, and discarding the data matrix in favor of data statistics, is preferable.
"""

import tensorflow as tf

class CovarianceMatrix(object):

    @classmethod
    def load_data(self, data):
        data = tf.constant(data, tf.float32)
        return CovarianceMatrix(covariance(data), column_norm(data))

    def __init__(self, covariance, column_norm):
        self.covariance = covariance
        self.column_norm = column_norm

    @tf.function(jit_compile = True)
    def naive_right_svd(self, subset_cov_batch):
        """Naive (brute-force) squared singular values, and V vectors.

        We never use platform SVD; we always use eigenvalues of a positive
        definite matrix (the singular values have been squared). The platform's
        (NumPy, TensorFlow) eigh function (Hermitian eigenvalues) should be in
        ascending order.
        """
        cov_lookup_left = subset_cov_batch[:, :, None, None]
        cov_lookup_right = subset_cov_batch[:, None, :, None]
        cov_lookup_left = cov_lookup_left + 0 * cov_lookup_right
        cov_lookup_right = cov_lookup_right + 0 * cov_lookup_left
        cov_lookup = tf.concat([cov_lookup_left, cov_lookup_right], axis=3)
        cov_batch_values = tf.gather_nd(self.covariance, cov_lookup)
        return tf.linalg.eigh(cov_batch_values)

def covariance(data_table):
    n_obs = data_table.shape[0]
    return tf.linalg.matmul(data_table, data_table, adjoint_a=True) / (
        n_obs - 1
    )

def column_norm(data_table):
    return tf.linalg.norm(data_table, axis=0)