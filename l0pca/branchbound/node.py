import tensorflow as tf

import bounds
import model
import partial_solution

def build_root(spca):
    y = -1 * tf.ones(tf.shape(spca.cov)[0], tf.int32)
    v = partial_solution.solve_pseudovec(spca, y)
    return y, bounds.select_next_branch_proj(v ** 2, y)

@tf.function
def process_node(spca, y, branch_node):
    y_1, contribution, y_1_bounds = bounds.bound_spca(
        spca,
        tf.tensor_scatter_nd_update(
            y,
            [[branch_node]],
            [0],
        ),
    )
    y_2, contribution, y_2_bounds = bounds.bound_spca(
        spca,
        tf.tensor_scatter_nd_update(
            y,
            [[branch_node]],
            [1],
        ),
    )

    if y_1_bounds[0] > y_2_bounds[0]:
        best_y = y_1_bounds[0]
        best_y_instance = y_1
    else:
        best_y = y_2_bounds[0]
        best_y_instance = y_2

    return y_1, y_1_bounds, contribution, y_2, y_2_bounds, contribution, best_y_instance, best_y