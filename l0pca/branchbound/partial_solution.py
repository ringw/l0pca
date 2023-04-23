import tensorflow as tf

import indexing
import model

def solve_with_positive_variables(spca, variables):
    sub_cov = tf.map_fn(lambda var_array: indexing.index_symmetric(spca.cov, var_array), variables, dtype=model.DTYPE)
    return tf.linalg.eigh(sub_cov)

def solve_pseudovec(spca, y):
    # What is the number of variables lower-bounded (forced on) for each y?
    y_lower_count = tf.math.reduce_sum(tf.cast(y == 1, tf.int32), axis=1)
    pseudovectors = tf.zeros_like(y, model.DTYPE)
    for i in tf.range(spca.k):
        y_locations = tf.squeeze(tf.where(y_lower_count == i), axis=1)
        y_i = tf.gather(y, y_locations)
        num_y_i = tf.shape(y_locations)[0]
        if i >= 2:
            # The evec is an optimal, unit-normed linear combination of i variables
            # to maximize sum of variance of the i variables in that direction.
            # Multiply through by sqrt cov in order to synthesize an observation
            # column which looks like some of the existing variables in this sqrt
            # matrix. Multiply through by sqrt cov again in order to take the dot
            # product of each variable with this synthetic variable.
            variables = tf.reshape(tf.where(y_i == 1)[:, 1], [num_y_i, i])
            _, evecs = solve_with_positive_variables(spca, variables)
            covariance_matrices_i = tf.map_fn(
                lambda var_array: indexing.index_columns(spca.cov, var_array),
                variables,
                dtype=model.DTYPE)
            vectors_i = tf.squeeze(covariance_matrices_i @ evecs[:, :, -1:], axis=2)
        else:
            # The eigvec is a linear combination of variables. Multiply through by
            # sqrt cov (which is given by multiplying by the corresponding
            # sqrt eigval), to get the synthesis of the variables. 
            vectors_i = tf.repeat(
                [spca.eigvec * tf.sqrt(spca.eigval)],
                num_y_i,
                axis=0,
            )
        pseudovectors = tf.tensor_scatter_nd_update(
            pseudovectors,
            y_locations[:, None],
            vectors_i,
        )
    return pseudovectors
