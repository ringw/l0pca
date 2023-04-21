import tensorflow as tf

import partial_solution
import top_k

def bound_spca(spca, y):
    u = partial_solution.solve_pseudovec(spca, y)
    projection_lb_vars = tf.math.reduce_sum(
        u ** 2 * tf.cast(y == 1, u.dtype)
    )
    projection_lb_stillneed = tf.math.reduce_sum(
        tf.math.top_k(
            u ** 2 * tf.cast(y == -1, u.dtype),
            spca.k - tf.math.reduce_sum(tf.cast(y == 1, tf.int32)),
        )[0]
    )
    projection_lb = projection_lb_vars + projection_lb_stillneed

    gershgorin_ub = tf.math.reduce_max(top_k.top_k(spca.cov_abs, spca.cov_perm, spca.k, y))
    frobenius_rows = top_k.top_k(spca.cov_2, spca.cov_perm, spca.k, y)
    frobenius_ub = tf.math.sqrt(tf.math.reduce_sum(
        tf.math.top_k(
            frobenius_rows,
            spca.k,
        )[0],
    ))
    return {
        'lower': projection_lb,
        'upper': tf.math.minimum(gershgorin_ub, frobenius_ub),
        'lower_spectral_contribution': u ** 2,
        'upper_sq_contribution': frobenius_rows,
    }