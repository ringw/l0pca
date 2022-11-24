"""PCA variable selection strategy.

NOTE: For now, there is no configurable strategy. However, for variable
selection (subset variables before truncated SVD, eigenvalues > 1), we might
want to take the trace of a sub-matrix (sum of top D eigenvalues). The strategy
would control both extracting the configured number of eigenvalues, and
implementing the lower/upper bound on sum of D eigenvalues for Optimal PCA.
"""

import tensorflow as tf

class PCAStrategy(object):

    @tf.function
    def evaluate_eigenvalues(self, eigenvalues):
        return eigenvalues[..., -1]