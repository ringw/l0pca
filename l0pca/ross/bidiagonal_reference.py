from l0pca.ross import bidiagonal_module
import numpy as np

DTYPE = np.float32
BATCH_DIM = 0
MATRIX_ROW_DIM = 1
MATRIX_COLUMN_DIM = 2

def sparse_to_dense(sparse_packed):
    num_arrays = sparse_packed.shape[0]
    if len(sparse_packed.shape) != 3 or sparse_packed.shape[1] != 3:
        raise ValueError('Invalid sparse data: {}'.format(sparse_packed.shape))
    n = sparse_packed.shape[2]
    dense = np.zeros((num_arrays, n, n), DTYPE)
    dense[:, range(n), range(n)] = sparse_packed[:, 0, :]
    dense[:, range(1, n), range(n - 1)] = sparse_packed[:, 1, 1:]
    dense[:, 0:-1, -1] = sparse_packed[:, 2, :-1]
    return dense

def dense_to_sparse(matrices):
    num_arrays = matrices.shape[0]
    n = matrices.shape[1]
    sparse_packed = np.zeros((num_arrays, 3, n), DTYPE)
    sparse_packed[:, 0, :] = matrices[:, range(n), range(n)]
    sparse_packed[:, 1, 1:] = matrices[:, range(1, n), range(n - 1)]
    sparse_packed[:, 2, :-1] = matrices[:, 0:-1, -1]
    return sparse_packed

def undo_rotation_atan2(x1, x2):
    """Returns an LHS rotation matrix to zero out the column entry at x1."""
    theta = np.arctan2(x1, x2)
    theta += np.where(
        theta > np.pi / 2.,
        -np.pi,
        np.where(
            theta < -np.pi / 2.,
            np.pi,
            0.))
    result = np.asarray(
        [[np.cos(theta), -np.sin(theta)],
         [np.sin(theta), np.cos(theta)]])
    return np.moveaxis(result, 2, 0)

def matmul(batch_a, batch_b):
    return np.einsum('ijk,ikl->ijl', batch_a, batch_b)

def update_augmented(matrix, row_index):
    rotation = undo_rotation_atan2(matrix[:, row_index, -1], matrix[:, row_index + 1, -1])
    full_matrix = np.eye(matrix.shape[MATRIX_ROW_DIM])
    full_matrix = np.repeat(
        full_matrix[None, :, :],
        matrix.shape[BATCH_DIM],
        axis=0)
    full_matrix[:, row_index:row_index+2, row_index:row_index+2] = rotation
    matrix = matmul(full_matrix, matrix)
    return matrix

def step_temp_cell(matrix, row_index, column_index):
    rotation = undo_rotation_atan2(-matrix[:, row_index, column_index], matrix[:, row_index, column_index + 1])
    full_matrix = np.eye(matrix.shape[MATRIX_ROW_DIM])
    full_matrix = np.repeat(
        full_matrix[None, :, :],
        matrix.shape[BATCH_DIM],
        axis=0)
    full_matrix[:, column_index:column_index+2, column_index:column_index+2] = rotation
    matrix = matmul(matrix, full_matrix)

    row_index -= 2
    column_index += 1
    rotation = undo_rotation_atan2(matrix[:, row_index, column_index], matrix[:, row_index + 1, column_index])
    full_matrix = np.eye(matrix.shape[MATRIX_ROW_DIM])
    full_matrix = np.repeat(
        full_matrix[None, :, :],
        matrix.shape[BATCH_DIM],
        axis=0)
    full_matrix[:, row_index:row_index+2, row_index:row_index+2] = rotation
    matrix = matmul(full_matrix, matrix)
    return matrix

def terminate_temp_value(matrix):
    row_index = 0
    rotation2 = undo_rotation_atan2(-matrix[:, row_index + 1, row_index], matrix[:, row_index + 1, row_index + 1])
    full_matrix_c = np.eye(matrix.shape[MATRIX_ROW_DIM])
    full_matrix_c = np.repeat(
        full_matrix_c[None, :, :],
        matrix.shape[BATCH_DIM],
        axis=0)
    full_matrix_c[:, row_index:row_index+2, row_index:row_index+2] = rotation2
    matrix = matmul(matrix, full_matrix_c)
    return matrix

class ReferenceModule(bidiagonal_module.BidiagonalModule):
    def update_augmented(self, sparse_packed, row_index):
        mat = update_augmented(sparse_to_dense(sparse_packed), row_index)
        temp_cell = bidiagonal_module.TempCell(mat[:, row_index + 1, row_index], row_index=row_index + 1, column_index=row_index)
        return dense_to_sparse(mat), temp_cell

    def move_temp_cell(sparse_packed, temp_cell):
        raise ValueError('Not implemented')