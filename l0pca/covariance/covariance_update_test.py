from l0pca.covariance import covariance_update
import numpy as np
import unittest

class CovarianceUpdateTest(unittest.TestCase):

    def setUp(self) -> None:
        np.random.seed()
        return super().setUp()

    def test_update_vector(self):
        np.random.seed(5)
        data_matrix = np.random.poisson(
            lam=20, size=(100, 10)).astype(np.float64)
        data_matrix -= data_matrix.mean(axis=0)[None, :]
        norms = np.linalg.norm(data_matrix, axis=0)
        orig_entries = np.asarray([3, 5, 7, 9, 1])
        update_entry = 8
        orig_cov = np.cov(data_matrix.T[orig_entries])
        update_cov = np.cov(data_matrix.T[list(orig_entries) + [update_entry]])

        aug_cov_matrix = np.zeros([6, 6])
        aug_cov_matrix[:5, :5] = orig_cov
        aug_cov_eigvals = np.zeros(6)
        aug_cov_eigvals[:5] = np.linalg.eigh(orig_cov)[0]
        aug_cov_eigvecs = np.zeros([6, 6])
        aug_cov_eigvecs[:5, :5] = np.linalg.eigh(orig_cov)[1]
        aug_cov_eigvecs[5, 5] = 1.
        aug_vec = covariance_update.update_vector(
            np.cov(data_matrix.T),
            orig_entries,
            np.linalg.eigh(orig_cov)[0],
            np.linalg.eigh(orig_cov)[1],
            update_entry,
        ).numpy()

        np.testing.assert_allclose(
            np.linalg.eigh(
                aug_cov_matrix + aug_vec[:, None].dot(aug_vec[None, :]))[0],
            np.linalg.eigh(update_cov)[0],
            # TODO: Fix the bug, these are incorrect values!
            atol=1.)

if __name__ == '__main__':
    unittest.main()