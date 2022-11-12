import numpy as np
from l0pca.bunch import pade
from l0pca.bunch import rank_one_eigenvalues
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

class RankOneEigenvaluesTest(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(4)
        self.counts = np.repeat(
            np.arange(5, 10),
            [8, 4, 4, 3, 1])

    def test_eigenvalue_initial_estimate(self):
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
        actual_vals = rank_one_eigenvalues.eigenvalue_initial_estimate(update_v, tf.math.sqrt(vals))
        np.testing.assert_almost_equal(actual_vals, expected_vals, decimal=3)

    def test_eigenvalue_initial_estimate_batch(self):
        data = np.random.normal(size=[7, 8, 100, 5])
        cov = np.zeros([7, 8, 4, 4])
        for i in range(cov.shape[0]):
            for j in range(cov.shape[1]):
                cov[i, j] = np.cov(data[i, j, :, 0:4].T)
        vals = np.linalg.eigh(cov)[0]
        # Use random fake vec instead of the real data column, for now.
        random_aug_data = np.random.uniform(size=[7, 8, 5])
        aug_vals_init = rank_one_eigenvalues.eigenvalue_initial_estimate(random_aug_data, np.sqrt(vals))

if __name__ == '__main__':
    unittest.main()