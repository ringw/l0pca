import tensorflow as tf

import bounds
import model
import partial_solution

def build_root(spca):
    y = -1 * tf.ones(tf.shape(spca.cov)[0], tf.int32)
    v = partial_solution.solve_pseudovec(spca, y)
    return y, bounds.select_next_branch_proj(v ** 2, y)

def apply_branch_node(y, branch_node):
    y = tf.ensure_shape(y, tf.TensorShape([None, None]))
    branch_node = tf.ensure_shape(branch_node, tf.TensorShape([None]))

    updates_shape = tf.shape(branch_node)
    # For each 1-D inner slice of y, spit out the location as a numeric
    # lookup.
    indices = tf.concat(
        [
            tf.range(tf.shape(y)[0])[:, None],
            branch_node[:, None],
        ],
        axis=-1,
    )
    y_1 = tf.tensor_scatter_nd_update(
        y,
        indices,
        tf.zeros(updates_shape, y.dtype),
    )
    y_2 = tf.tensor_scatter_nd_update(
        y,
        indices,
        tf.ones(updates_shape, y.dtype),
    )
    return y_1, y_2

def process_node(spca, y, branch_node):
    y_1, y_2 = apply_branch_node(y, branch_node)
    y_1, branch_node_1, y_1_bounds = bounds.bound_spca(
        spca,
        y_1,
    )
    y_2, branch_node_2, y_2_bounds = bounds.bound_spca(
        spca,
        y_2,
    )

    # Best lower bound from this round. From y, guessing the concrete k used for
    # the lower bound is completely deterministic, so we can store a node that
    # is not yet solved instead of explicitly passing a subset of size k.
    # However, it would also be prudent to encapsulate that and use tf.where()
    # here, producing a tf.int32 array of length k instead.
    best_y_1 = tf.math.argmax(y_1_bounds[:, 0])
    best_y_2 = tf.math.argmax(y_2_bounds[:, 0])
    if y_1_bounds[0, best_y_1] > y_2_bounds[0, best_y_2]:
        best_y = y_1_bounds[best_y_1, 0]
        best_y_instance = y_1[best_y_1, :]
    else:
        best_y = y_2_bounds[best_y_2, 0]
        best_y_instance = y_2[best_y_2, :]

    return y_1, y_1_bounds, branch_node_1, y_2, y_2_bounds, branch_node_2, best_y_instance, best_y