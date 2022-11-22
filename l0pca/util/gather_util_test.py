from l0pca.util import gather_util
import numpy as np
import tensorflow as tf
import unittest

class GatherUtilTest(unittest.TestCase):

    def test_repeat_gather(self):
        A = tf.reshape(
            tf.range(3 * 5 * 2 * 2),
            [3, 5, 2, 2],
        )
        # From a batch (0, 0), the elements at index (0, 0) and (1, 1) are 0 and 3.
        np.testing.assert_equal(
            np.asarray([0, 3]),
            gather_util.repeat_gather(
                A, [[0, 0], [1, 1]], batch_dims=2)[0, 0].numpy())

if __name__ == '__main__':
    unittest.main()