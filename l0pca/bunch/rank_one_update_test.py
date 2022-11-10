import numpy as np
from l0pca.bunch import pade
from l0pca.bunch import rank_one_update
import tensorflow as tf
import unittest

class SvdProblem(object):

    def __init__(self, data) -> None:
        data = data.copy() - data.mean(axis=0)
        self.data = data
        self.num_row = data.shape[0]
        self.feature_norm = np.linalg.norm(data, axis=0)

    def run_column_cov_ev(self, columns):
        return np.linalg.eigh(self.run_column_cov(columns))[0]

    def run_column_cov_evecs(self, columns):
        return np.linalg.eigh(self.run_column_cov(columns))[1]

    def run_column_cov(self, columns):
        subset = self.data[:, columns]
        return subset.T.dot(subset) / (self.num_row - 1)

    def compute_cov(self):
        return self.run_column_cov(np.arange(0, self.data.shape[1]))

class RankOneUpdateTest(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(4)
        self.counts = np.repeat(
            np.arange(5, 10),
            [8, 4, 4, 3, 1])
    
    def test_ev(self):
        prob = SvdProblem(np.random.poisson(self.counts[None, :], [100, 20]))
        # The max singular vector MUST use a linear combination of features and
        # produce a result explaining somewhat more variance than any one
        # feature.
        self.assertGreater(
            prob.run_column_cov_ev([4, 8, 15]).max(),
            prob.data.var(axis=0)[[4, 8, 15]].max())

    def test_update_matrix(self):
        prob = SvdProblem(np.random.poisson(self.counts[None, :], [100, 20]))
        vals = prob.run_column_cov_ev([2, 3])
        vecs = prob.run_column_cov_evecs([2, 3])
        expected_vals = prob.run_column_cov_ev([2, 3, 10])
        update_v = rank_one_update.create_update_append_column(
            np.asarray([2, 3]),
            np.sqrt(vals),
            vecs,
            10,
            prob.compute_cov(),
            prob.feature_norm,
            prob.num_row)
        guess = vals[0] + (vals[1] - vals[0]) * 0.5
        update_taylor = rank_one_update.bunch_rational_function_taylor_series(
            update_v, np.sqrt(vals), guess)
        actual_vals = tf.linalg.eigh(
            np.diag([0.] + list(vals))
            + tf.linalg.matmul(
                update_v[:, None], update_v[None, :]))[0]
        upd = pade.pade_zero_find(update_taylor)
        result = guess + upd
        np.testing.assert_almost_equal(actual_vals, expected_vals)

if __name__ == '__main__':
    unittest.main()