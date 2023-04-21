import tensorflow as tf

import indexing

def solve_with_positive_variables(spca, y):
    variables = tf.where(y == 1)[:, 0]
    sub_cov = indexing.index_symmetric(spca.cov, variables)
    evals, evecs = tf.linalg.eigh(sub_cov)
    return variables, evals, evecs

def solve_pseudovec(spca, y):
    variables = tf.where(y == 1)[:, 0]
    if variables.shape[0] >= 2:
        # The evec is an optimal, unit-normed linear combination of i variables
        # to maximize sum of variance of the i variables in that direction.
        # Multiply through by sqrt cov in order to synthesize an observation
        # column which looks like some of the existing variables in this sqrt
        # matrix. Multiply through by sqrt cov again in order to take the dot
        # product of each variable with this synthetic variable.
        _, _, evecs = solve_with_positive_variables(spca, y)
        return tf.squeeze(indexing.index_columns(spca.cov, variables) @ evecs[:, 0:1], axis=1)
    else:
        # The eigvec is a linear combination of variables. Multiply through by
        # sqrt cov (which is given by multiplying by the corresponding
        # sqrt eigval), to get the synthesis of the variables. 
        return spca.eigvec * tf.sqrt(spca.eigval)