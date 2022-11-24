import numpy as np
from l0pca.bruteforce import rankone_batches
import tensorflow as tf
import unittest

class RankoneBatchesTest(unittest.TestCase):

    def test_pool_combinations(self):
        fake_combinations = (
            # List of combinations, k = 0.
            tf.constant([[]]),
            # List of combinations, k = 1.
            tf.constant([[1], [5]]),
            # List of combinations, k = 2.
            tf.constant([[2, 9], [3, 10], [4, 11]]),
            # List of combinations, k = 3.
            tf.constant([[1, 4, 7]]),
        )
        pool_arr = tf.constant([[3, 2], [5, 1]])
        # Outer product: First 2 columns are our "list of combinations k = 2"
        # plus FEATURE_POOL_SIZE * 3. Second column is our k = 1 plus
        # FEATURE_POOL_SIZE * 5.
        np.testing.assert_array_equal(
            rankone_batches.pool_apply_prefix(pool_arr, fake_combinations),
            [[50, 57, 81],
             [50, 57, 85],
             [51, 58, 81],
             [51, 58, 85],
             [52, 59, 81],
             [52, 59, 85]],
        )

if __name__ == '__main__':
    unittest.main()