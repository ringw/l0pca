import numpy as np
from l0pca.inverse import singular_matrix
import unittest

class SingularMatrixTest(unittest.TestCase):

    def test_factory(self):
        A = np.zeros((2, 3, 3))
        A[0, [0, 1, 2], [0, 1, 2]] = [3., 4., 5.]
        A[0, :-1, -1] = [-1., 2.]
        A[1, [0, 1, 2], [0, 1, 2]] = [-2., 7., 8.]
        A[1, :-1, -1] = [0., -5.]
        mat = singular_matrix.SingularMatrix.from_dense(A)

if __name__ == '__main__':
    unittest.main()