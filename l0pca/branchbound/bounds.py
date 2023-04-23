import tensorflow as tf

import partial_solution
import top_k

def is_terminal(spca, y):
    return tf.logical_or(
        tf.math.reduce_sum(tf.cast(y != 0, tf.int32)) == spca.k,
        tf.math.reduce_sum(tf.cast(y == 1, tf.int32)) == spca.k,
    )

def apply_cut_spca(spca, y, constant_lb, projection_lb, frobenius_sq_rows_ub):
    contribution_sq_ub = 2*frobenius_sq_rows_ub - spca.variance
    contribution_sq_lb = constant_lb[..., None]**2 - (constant_lb[..., None]-projection_lb)**2
    return tf.where(
        (contribution_sq_ub < contribution_sq_lb) & (y == -1),
        0,
        y,
    )

def select_next_branch_proj(proj_trace, y):
    return tf.cast(tf.math.argmax(proj_trace * tf.cast(y == -1, proj_trace.dtype)), tf.int32)

def bound_spca(spca, y, cut_node=False):
    stillneed = spca.k - tf.math.reduce_sum(tf.cast(y == 1, tf.int32))
    # Rank-one projection of the Hermitian problem.
    proj_trace = partial_solution.solve_pseudovec(spca, y) ** 2
    projection_lb_vars = tf.math.reduce_sum(
        proj_trace * tf.cast(y == 1, proj_trace.dtype)
    )
    projection_lb_stillneed = tf.math.reduce_sum(
        tf.math.top_k(
            proj_trace * tf.cast(y == -1, proj_trace.dtype),
            stillneed,
        )[0]
    )
    projection_lb = projection_lb_vars + projection_lb_stillneed

    # We have just diagonalized the k by k problem. The projection_lb is
    # complete. It doesn't matter which node we return to "branch" on,
    # because there is no branching to be done.
    if is_terminal(spca, y):
        return y, tf.constant(0), tf.convert_to_tensor([projection_lb, projection_lb])

    trace_ub = tf.squeeze(
        tf.math.reduce_sum(spca.variance * tf.cast(y == 1, spca.variance.dtype))
            + tf.math.reduce_sum(tf.math.top_k(spca.variance * tf.cast(y == -1, spca.variance.dtype), stillneed)[0]),
    )
    gershgorin_ub = tf.math.reduce_max(top_k.top_k(spca.cov_abs, spca.cov_perm, spca.k, y))
    frobenius_rows = top_k.top_k(spca.cov_2, spca.cov_perm, spca.k, y)
    frobenius_ub = tf.math.sqrt(tf.math.reduce_sum(
        tf.math.top_k(
            frobenius_rows,
            spca.k,
        )[0],
    ))
    # TODO: Upper bounds can change after filtering out variables in the cut, so
    # we should have one more iteration of top_k.
    if cut_node:
        y = apply_cut_spca(spca, y, projection_lb, proj_trace, frobenius_rows)

    upper_bound = tf.math.reduce_min([trace_ub, gershgorin_ub, frobenius_ub], axis=0)
    bounds_array = tf.concat(
        [
            projection_lb[..., None],
            upper_bound[..., None],
        ],
        axis=-1,
    )
    return y, select_next_branch_proj(proj_trace, y), bounds_array