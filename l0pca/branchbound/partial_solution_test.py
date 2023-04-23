import numpy as np
import tensorflow as tf
import unittest

import model
import partial_solution

class PartialSolutionTest(unittest.TestCase):

    def test_diagonal_matrix(self):
        cov = tf.constant(
            [[2., 0., 0., 0.],
             [0., 1.5, 0., 0.],
             [0., 0., 1.2, 0.],
             [0., 0., 0., 1.]],
        )
        spca = model.Spca(cov, k=3)
        y = tf.constant(
            [[-1, -1, 0, 1],
             [1, 1, 0, -1]],
        )
        np.testing.assert_allclose(
            partial_solution.solve_pseudovec(spca, y).numpy(),
            np.asarray(
                [
                    [1., 0., 0., 0.],
                    [2. / np.sqrt(2**2 + 1.5**2), 1.5 / np.sqrt(2**2 + 1.5**2), 0., 0.],
                ]
            )
        )

if __name__ == '__main__':
    unittest.main()