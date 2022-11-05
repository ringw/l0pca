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
    # data_11: First diagonal affected.
    data_11 = packed[:, 0, row_index]
    # data_12: Off-diagonal (close to zero after previous steps).
    data_12 = packed[:, 1, row_index]
    # data_13: Augmented entry (concatenate this column which is not actually
    # contiguous to the other columns).
    data_13 = packed[:, 2, row_index]
    data_21 = tf.zeros_like(data_11)
    data_22 = packed[:, 0, row_index + 1]
    # data_23: Augmented entry (remember not contiguous to data_22).
    data_23 = packed[:, 2, row_index + 1]
    # Transpose (reverse all dims), so initally column-major order.
    perturbed_dense = tf.convert_to_tensor(
        [
            [data_11, data_21],
            [data_12, data_22],
            [data_13, data_23],
        ]
    )
    perturbed_dense = tf.transpose(perturbed_dense)
    perturbed_dense = tf.linalg.matmul(rotate_cols, perturbed_dense)

    packed = update_packed_rows(
        packed,
        row_index,
        diag=[perturbed_dense[:, 0, 0], perturbed_dense[:, 1, 1]],
        off_diag=[perturbed_dense[:, 0, 1], tf.zeros_like(data_11)],
        aug=[perturbed_dense[:, 0, 2], perturbed_dense[:, 1, 2]],
    )
    temp_value = perturbed_dense[:, 1, 0]
    temp_row = row_index + 1
    temp_column = row_index
    return packed, temp_value, temp_row, temp_column

@tf.function(jit_compile=True)
def step_temp_cell(packed, temp_value, row_index, column_index):
    # Precondition: column_index > 0.
    # Precondition: column_index + 1 == row_index.
    # The temp-value is at the bottom-center of a 3x3 tile, and this
    # off-diagonal value will be moved to the center-left using two
    # rotations.
    # data_11: Previous diagonal.
    data_11 = packed[:, 0, column_index - 1]
    data_12 = packed[:, 1, column_index - 1]
    data_13 = tf.zeros_like(data_11)
    data_21 = tf.zeros_like(data_11)
    data_22 = packed[:, 0, column_index]
    data_23 = packed[:, 1, column_index]
    data_31 = tf.zeros_like(data_11)
    data_32 = temp_value
    data_33 = packed[:, 0, column_index + 1]
    perturbed_dense = tf.convert_to_tensor(
        [
            [data_11, data_21, data_31],
            [data_12, data_22, data_32],
            [data_13, data_23, data_33],
        ]
    )
    perturbed_dense = tf.transpose(perturbed_dense)

    rotate_rows = undo_rotation_atan2(-data_32, data_33)
    rotate_rows = tf.concat(
        [
            tf.concat(
                [
                    tf.ones_like(rotate_rows[:, 0:1, 0:1]),
                    tf.zeros_like(rotate_rows[:, 0:1, :]),
                ],
                axis=-1,
            ),
            tf.concat(
                [tf.zeros_like(rotate_rows[:, :, 0:1]), rotate_rows],
                axis=-1),
        ],
        axis=-2,
    )
    perturbed_dense = tf.linalg.matmul(perturbed_dense, rotate_rows)

    rotate_columns = undo_rotation_atan2(
        perturbed_dense[:, 0, 2],
        perturbed_dense[:, 1, 2],
    )
    rotate_columns = tf.concat(
        [
            tf.concat(
                [
                    rotate_columns,
                    tf.zeros_like(rotate_columns[:, :, 0:1]),
                ],
                axis=-1,
            ),
            tf.concat(
                [
                    tf.zeros_like(rotate_columns[:, 0:1, :]),
                    tf.ones_like(rotate_columns[:, 0:1, 0:1]),
                ],
                axis=-1,
            ),
        ],
        axis=-2,
    )
    perturbed_dense = tf.linalg.matmul(rotate_columns, perturbed_dense)

    packed = update_packed_rows(
        packed,
        column_index - 1,
        diag=[perturbed_dense[:, 0, 0], perturbed_dense[:, 1, 1], perturbed_dense[:, 2, 2]],
        off_diag=[perturbed_dense[:, 0, 1], perturbed_dense[:, 1, 2], packed[:, 1, column_index + 1]],
        # aug was unchanged.
        aug=[packed[:, 2, column_index - 1], packed[:, 2, column_index], packed[:, 2, column_index + 1]],
    )
    temp_value = perturbed_dense[:, 1, 0]
    return packed, temp_value, row_index - 1, column_index - 1

@tf.function(jit_compile=True)
def terminate_step(packed, temp_value):
    # Precondition: temp_row == 1.
    # Precondition: temp_column == 0.
    # data_11: First diagonal.
    data_11 = packed[:, 0, 0]
    data_12 = packed[:, 1, 0]
    data_21 = temp_value
    data_22 = packed[:, 0, 1]
    perturbed_dense = tf.convert_to_tensor(
        [[data_11, data_21], [data_12, data_22]]
    )
    perturbed_dense = tf.transpose(perturbed_dense)

    rotate_rows = undo_rotation_atan2(-data_21, data_22)
    perturbed_dense = tf.linalg.matmul(perturbed_dense, rotate_rows)

    return update_packed_rows(
        packed,
        row_index=0,
        diag=[perturbed_dense[:, 0, 0], perturbed_dense[:, 1, 1]],
        off_diag=[perturbed_dense[:, 0, 1], packed[:, 1, 1]],
        aug=[packed[:, 2, 0], packed[:, 2, 1]],
    )

@tf.function
def diagonalize_packed_storage(packed):
    packed_shape = packed.shape
    n = packed.shape[-1]
    def _major_iteration(packed, i):
        packed, temp_value, temp_row, temp_column = update_augmented(packed, i)
        def _minor_iteration(packed, temp_value, temp_row, temp_column):
            packed, temp_value, temp_row, temp_column = step_temp_cell(packed, temp_value, temp_row, temp_column)
            return packed, temp_value, temp_row, temp_column
        packed, temp_value, _, _ = tf.while_loop(
            # While temp_column > 1, then reduce the value of temp_column.
            lambda packed, temp_value, temp_row, temp_column: temp_column > 1,
            _minor_iteration,
            (packed, temp_value, temp_row, temp_column),
            maximum_iterations=i,
        )
        packed = terminate_step(packed, temp_value)
        del temp_value
        packed = tf.ensure_shape(packed, packed_shape)
        return packed, i + 1
    return tf.while_loop(
        lambda *args: True,
        _major_iteration,
        (packed, 0),
        # TODO: Test the last few entries in augmented column. Should increase to n - 1.
        maximum_iterations=n - 2,
    )

class PackedModule(bidiagonal_module.BidiagonalModule):
    def update_augmented(self, sparse_packed, row_index):
        mat, temp_value, temp_row, temp_column = update_augmented(sparse_packed, row_index)
        return mat, bidiagonal_module.TempCell(temp_value, temp_row, temp_column)

    def move_temp_cell(self, sparse_packed, temp_cell):
        raise ValueError('Not implemented')