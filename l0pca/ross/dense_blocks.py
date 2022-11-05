import numpy as np

DTYPE = np.float32

def new_random_matrices(num_batch, num_rows):
    values = np.random.random((num_batch, num_rows + (num_rows - 1))).astype(DTYPE)
    packed_storage = np.zeros((num_batch, 3, num_rows), DTYPE)
    packed_storage[:, 0, :] = values[:, 0:num_rows]
    # Original matrices do not have an off-diagonal.
    packed_storage[:, 2, :-1] = values[:, num_rows:]
    return packed_storage

def pack_ross_matrix(singular_values, augmented_values, orthogonal_value):
    num_rows = singular_values.shape[-1] + 1
    if len(singular_values.shape) == 1:
        singular_values = singular_values[None, :]
        augmented_values = augmented_values[None, :]
        orthogonal_value = orthogonal_value[None]
    num_batch = singular_values.shape[0]
    packed_storage = np.zeros((num_batch, 3, num_rows), DTYPE)
    packed_storage[:, 0, :-1] = singular_values
    packed_storage[:, 0, -1] = orthogonal_value
    # Original matrices do not have an off-diagonal.
    packed_storage[:, 2, :-1] = augmented_values
    return packed_storage