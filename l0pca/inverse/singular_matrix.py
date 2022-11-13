"""Sparse matrix class for updating singular values (Ross 2008)."""

import numpy as np
import tensorflow as tf

class SingularMatrix(object):

    @classmethod
    def from_dense(self, matrices):
        batch_shape = list(matrices.shape[:-2])
        n = matrices.shape[-1]
        diagonal = np.full(batch_shape + [n], np.nan)
        aug_column = np.full(batch_shape + [n - 1], np.nan)
        matrix_lookup = list(map(tuple, zip(*np.where(np.ones_like(matrices[:, 0, 0])))))
        for lookup_inds in matrix_lookup:
            matrix = matrices[lookup_inds]
            # Squeeze matrix to 2D.
            # matrix = np.squeeze(matrix, tuple(range(len(lookup_inds))))
            for i in range(n):
                for j in range(n):
                    if i == j:
                        diagonal[lookup_inds + tuple([i])] = matrix[i, j]
                    elif j == n - 1:
                        aug_column[lookup_inds + tuple([i])] = matrix[i, j]
                    elif matrix[i, j] != 0.:
                        raise ValueError('Non-sparse matrix at {}'.format((i, j)))
        return SingularMatrix(diagonal, aug_column)

    def __init__(self, diagonal, aug_column):
        self.diagonal = diagonal
        self.aug_column = aug_column

    @tf.function
    def dot(self, mat):
        result = self.diagonal[..., :, None] * mat
        result += tf.linalg.matmul(self.aug_column[..., :, None], mat[..., -1:, :])
        return result

    @tf.function
    def dot_as_adjoint(self, mat):
        pass

    @tf.function
    def dot_constant(self, constant):
        return SingularMatrix(self.diagonal * constant, self.aug_column * constant)

    @tf.function
    def invert(self):
        # A = diagonal[:-1]
        # B = aug_column
        # C = 0
        # D = diagonal[-1:]
        A_inv = 1. / self.diagonal[..., :-1]
        B_inv = -1. / self.diagonal[..., :-1] * self.aug_column / self.diagonal[..., -1:]
        D_inv = 1. / self.diagonal[..., -1:]
        return SingularMatrix(
            tf.concat([A_inv, D_inv], axis=-1),
            B_inv)