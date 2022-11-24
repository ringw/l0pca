"""Batching combinatorics utility.

This decomposes the combinatorics problem arbitrarily, into an outer loop that
can be interpreted at high-level here, and an inner loop (few number of selected
variables). The purpose of the outer loop is to partition the graph
(pre-selecting elements in the subset up to a certain position in the array).
Each "pool" of features up to that point chosen here is considered already
decided (as a tree structure), and leaves some trailing elements to be processed
in parallel using rank-one updates.
"""
import itertools
import numpy as np
import tensorflow as tf

# Consider: 16 choose 3 ~= 512, a useful number of cores on GPU. Brute-force
# k will not be much greater than 3 (for now, hardcode 3 <= k <= 5). Platform's
# SVD will be based on PREFIX_K number of features from the first 16 features,
# and we can precompute each one eagerly. We can afford the dense matrices, as
# well as a DFS stack to compute every rank-one update up to the full K. Now, if
# the graph is searched using 
FEATURE_POOL_SIZE = 16
PREFIX_K_SIZE = 3

def initialize_batches(n):
    pool_edge = np.arange(0, n - FEATURE_POOL_SIZE, FEATURE_POOL_SIZE)
    # Note: This is double-counting some assignments of the multiset of pools.
    num_batches = len(pool_edge) ** PREFIX_K_SIZE
    batch_num = np.arange(num_batches)
    for batch_num in range(num_batches):
        pool_ident = []
        batch_lookup = batch_num
        for _ in range(PREFIX_K_SIZE):
            pool_ident.append(batch_lookup % len(pool_edge))
            batch_lookup //= len(pool_edge)
        pool_arr = [(pool_ident[0], 1)]
        if pool_ident[1] == pool_ident[0]:
            pool_arr[0] = (pool_ident[0], 2)
        else:
            pool_arr.append((pool_ident[1], 1))
        
        if pool_ident[2] == pool_ident[0]:
            pool_arr[0] = (pool_ident[0], pool_arr[0][1] + 1)
        elif len(pool_arr) > 1 and pool_ident[2] == pool_ident[1]:
            pool_arr[1] = (pool_ident[1], pool_arr[1][1] + 1)
        else:
            pool_arr.append((pool_ident[2], 1))
        pool_arr = sorted(pool_arr, key=lambda tup: tup[0])

        yield pool_arr

def initialize_combinations_data():
    """Initialize combinatorial TF constant arrays."""
    def combinations(n, k):
        return np.asarray(list(itertools.combinations(list(range(n)), k)))
    return tuple(
        combinations(FEATURE_POOL_SIZE, i) for i in range(PREFIX_K_SIZE + 1)
    )

def pool_apply_prefix(pool_tensor, comb_tensor):
    """TF function to generate k=3 combinations.

    This is an inner loop (already batched k=3 combinations arbitrarily, based
    on chunking/pooling data matrix). Therefore, at this point, we should use
    tf.function and not vanilla Python (itertools).
    """
    pool_result = comb_tensor[pool_tensor[0, 1]] + pool_tensor[0, 0] * FEATURE_POOL_SIZE
    if pool_tensor.shape[0] >= 2:
        next_pool = comb_tensor[pool_tensor[1, 1]] + pool_tensor[1, 0] * FEATURE_POOL_SIZE
        # Broadcast the first and second (outer product dims), DO NOT broadcast
        # the last dim because different pools have different num elements.
        pool_result = pool_result[:, None, :] + 0 * next_pool[None, :, 0:1]
        next_pool = next_pool[None, :, :] + 0 * pool_result[:, :, 0:1]
        pool_result = tf.concat([pool_result, next_pool], axis=2)
        pool_result = tf.reshape(pool_result, [-1, pool_tensor[0, 1] + pool_tensor[1, 1]])
    if pool_tensor.shape[0] >= 3:
        raise NotImplementedError()
    return pool_result