from l0pca.covariance import matrix
import numpy as np
import unittest

def generate_data(seed=4):
    np.random.seed(seed)
    data = np.random.poisson(lam=10, size=(100, 20))
    data = data.astype(np.float32)
    data -= data.mean(axis=0)[None, :]
    return data

def naive_cov_matrix(data, columns):
    return np.cov(data[:, columns].T)

class CovarianceMatrixTest(unittest.TestCase):

    def setUp(self) -> None:
        np.random.seed()
        return super().setUp()

    def test_naive_right_svd(self):
        data = generate_data()
        cov = matrix.CovarianceMatrix.load_data(data)
        columns = [4, 6, 8]
        eigvals, eigvecs = cov.naive_right_svd(np.asarray([columns]))
        np.testing.assert_allclose(
            eigvals[0, :],
            np.linalg.eigh(naive_cov_matrix(data, columns))[0],
            rtol=1e-6,
        )
        expected_evecs = np.linalg.eigh(naive_cov_matrix(data, columns))[1]
        # Sign multiplier of each column is impl-dependent.
        eigvecs *= (np.sign(eigvecs[0, 0, :]) * np.sign(expected_evecs[0, :]))[None, None, :]
        np.testing.assert_allclose(
            eigvecs[0, :],
            expected_evecs,
            rtol=1e-5,
        )
 
if __name__ == '__main__':
    unittest.main()