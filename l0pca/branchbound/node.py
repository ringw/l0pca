import tensorflow as tf

import bounds
import model
import partial_solution

def build_root(spca):
    y = -1 * tf.ones(tf.shape(spca.cov)[0], tf.int32)
    u = partial_solution.solve_pseudovec(spca, y)
    return tf.concat(
        [
            tf.cast(y, u.dtype)[:, None],
            (u ** 2)[:, None],
        ],
        axis=1,
    )

def process_node(spca, node):
    y = tf.cast(node[:, 0], tf.int32)
    contribution = node[:, 1]
    available_var = tf.where(y == -1)[:, 0]
    branch_node = available_var[
        tf.math.argmax(tf.gather(contribution, available_var))
    ]

    y_1 = tf.tensor_scatter_nd_update(
        y,
        [[branch_node]],
        [0],
    )
    # TODO: Trimming y_1 may allow for tighter bounds, so re-run bounds for a
    # second iteration.
    y_1_bounds = bounds.bound_spca(spca, y_1)
    y_2 = tf.tensor_scatter_nd_update(
        y,
        [[branch_node]],
        [1],
    )
    y_2_bounds = bounds.bound_spca(spca, y_2)

    terminate_best_y = tf.cond(
        y_1_bounds['lower'] > y_2_bounds['lower'],
        lambda: y_1,
        lambda: y_2,
    )
    if y_1_bounds['lower'] > y_2_bounds['lower']:
        best_y = y_1_bounds['lower']
        best_y_instance = y_1
    else:
        best_y = y_2_bounds['lower']
        best_y_instance = y_2

    y_1 = tf.concat(
        [
            tf.cast(y_1, model.DTYPE)[:, None],
            y_1_bounds['lower_spectral_contribution'][:, None],
        ],
        axis=1,
    )
    y_2 = tf.concat(
        [
            tf.cast(y_2, model.DTYPE)[:, None],
            y_2_bounds['lower_spectral_contribution'][:, None],
        ],
        axis=1,
    )

    return y_1_bounds['lower'], y_1_bounds['upper'], y_1, y_2_bounds['lower'], y_2_bounds['upper'], y_2, best_y, best_y_instance