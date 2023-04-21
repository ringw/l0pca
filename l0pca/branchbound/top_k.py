import tensorflow as tf

import indexing

def top_k(M, order, k, y):
    n = M.shape[0]
    variables = tf.where(y == 1)[:, 0]
    starting_sums = tf.math.reduce_sum(indexing.index_columns(M, variables), axis=1)

    # If y == 1, then we will double-count including startingsums 
    stillneed = tf.constant(k) - tf.shape(variables)[0]
    stillneed = tf.where(y == 1, stillneed, stillneed - 1)[:, None]
    sorted_status = tf.gather(y, order)
    count_variables = sorted_status == -1
    ranking = tf.cumsum(tf.cast(count_variables, tf.int32), axis=1)
    weight_lookup = (
        tf.range(1, 1+M.shape[0], dtype=tf.int32)[:, None] *
        # Note: ranking for first available in y will start at a value of 1.
        tf.cast((ranking <= stillneed)
        & (count_variables | ((y[:, None] == -1) & (order == tf.range(n)[:, None]))), tf.int32)
    )
    available_y = tf.math.bincount(weight_lookup, M, minlength=n+1, maxlength=n+1)[1:]

    return starting_sums + available_y