import numpy as np
from l0pca.rankone import rankone_reference
from l0pca.rankone import secular_equation
import unittest

class SecularEquationTest(unittest.TestCase):

    def test_small_matrix(self):
        original_eig = np.asarray(
            [1.2611, 1.478, 1.4786, 2.6058, 2.9202]
        )
        update_vec = np.asarray([0.3175, -0.1634, 0.663, -0.3737, 0.8327])
        initial_eig = np.zeros_like(original_eig)
        initial_eig[0] = (original_eig[0] + original_eig[1]) / 2
        initial_eig[1] = (original_eig[1] + original_eig[2]) / 2
        initial_eig[2] = (original_eig[2] + original_eig[3]) / 2
        initial_eig[3] = (original_eig[3] + original_eig[4]) / 2
        initial_eig[4] = original_eig[4] + (np.linalg.norm(update_vec) ** 2 / 2)
        np.testing.assert_allclose(
            secular_equation.eq_search_eigenvalues_iteration(
                original_eig[None, None, :],
                update_vec[None, None, :],
                initial_eig[None, None, :]),
            rankone_reference.rankone_update_eigenvalues(original_eig, update_vec)[
                None, None, :
            ],
            rtol=0.005)

if __name__ == '__main__':
    unittest.main()