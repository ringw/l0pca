from l0pca.ross import bidiagonal_module
import numpy as np
import tensorflow as tf

@tf.function
def new_batched_matrix_tensor(arr):
    return tf.transpose(
        tf.convert_to_tensor(arr),
        [2, 0, 1],
    )

@tf.function
def undo_rotation_atan2(x1, x2):
    """Returns an LHS rotation matrix theta to zero out the column entry at x1."""
    theta = tf.math.atan2(x1, x2)
    pi = tf.constant(np.pi, tf.float32)
    pi2 = tf.constant(np.pi / 2., tf.float32)
    theta += tf.where(
        theta > pi2,
        -pi,
        tf.where(
            theta < -pi2,
            pi,
            0.))
    result = tf.convert_to_tensor(
        [[tf.math.cos(theta), -tf.math.sin(theta)],
         [tf.math.sin(theta), tf.math.cos(theta)]])
    return tf.transpose(result, [2, 0, 1])

@tf.function
def update_packed_rows(packed, row_index, diag, off_diag, aug):
    # Careful! Reverse every dimension of a multi-dimensional array. The shape
    # is now (n_row, 3, n_batch), so we can concat a row slice easily.
    row_slice_packed = tf.transpose(packed)
    row_slice_before = row_slice_packed[:row_index]
    num_slices = len(diag)
    row_slice_after = row_slice_packed[row_index + num_slices:]
    insert_list = []
    for i in range(num_slices):
        insert_row = [
            diag[i],
            off_diag[i],
            aug[i],
        ]
        insert_list.append(insert_row)
    insert_slice = tf.convert_to_tensor(insert_list)
    return tf.transpose(
        tf.concat(
            [
                row_slice_before,
                insert_slice,
                row_slice_after,
            ],
            axis=0,
        )
    )

@tf.function(jit_compile = True)
def update_augmented(packed, row_index):
    aug1 = packed[:, 2, row_index]
    aug2 = packed[:, 2, row_index + 1]
    rotate_cols = undo_rotation_atan2(aug1, aug2)
    data_11 = packed[:, 1, row_index - 1]
    data_11 = data_11 if row_index > 0 else tf.zeros_like(data_11)
    perturbed_dense = tf.convert_to_tensor(
        [
            [
                # data_11: It is the previous off-diagonal, or may be out of bounds.
                data_11,
                # data_21: This will be our new temp cell, currently sparse.
                tf.zeros_like(data_11),
            ],
            [
                # data_12: 
                packed[:, 0, row_index],
                # data_
                packed[:, 1, row_index],
            ],
            [
                tf.zeros_like(data_11),
                packed[:, 0, row_index + 1],
            ],
            [
                packed[:, 2, row_index],
                packed[:, 2, row_index + 1],
            ],
        ]
    )
    perturbed_dense = tf.transpose(perturbed_dense)
    perturbed_dense = tf.linalg.matmul(rotate_cols, perturbed_dense)

    # Now rotate on the RHS...

    # Set some fake data so that we can test the throughput into and out of our packed sparse rows.
    packed = update_packed_rows(
        packed,
        row_index,
        [perturbed_dense[:, 0, 0], perturbed_dense[:, 1, 1]],
        [perturbed_dense[:, 0, 1], perturbed_dense[:, 1, 2]],
        [tf.zeros_like(aug1), perturbed_dense[:, 1, 0]],
    )
    temp_value = perturbed_dense[:, 0, 2]
    temp_row = row_index - 1
    temp_column = row_index + 1
    return packed, temp_value, temp_row, temp_column

class PackedModule(bidiagonal_module.BidiagonalModule):
    def update_augmented(self, sparse_packed, row_index):
        mat, temp_value, temp_row, temp_column = update_augmented(sparse_packed, row_index)
        return mat, bidiagonal_module.TempCell(temp_value, temp_row, temp_column)

    def move_temp_cell(self, sparse_packed, temp_cell):
        raise ValueError('Not implemented')