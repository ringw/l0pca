import numpy as np
import tensorflow as tf
import unittest

import node

class NodeTest(unittest.TestCase):

    def test_apply_branch_node_single_update(self):
        y_1, y_2 = node.apply_branch_node(tf.constant([[-1, -1, 1]]), tf.constant([1]))
        np.testing.assert_equal(y_1.numpy(), [[-1, 0, 1]])
        np.testing.assert_equal(y_2.numpy(), [[-1, 1, 1]])

    def test_apply_branch_node(self):
        y = tf.constant(
            [[-1, -1, 0, -1, -1],
             [0, -1, 0, -1, 1]])
        branch_node = tf.constant([0, 1])
        y_1, y_2 = node.apply_branch_node(y, branch_node)

        np.testing.assert_equal(
            y_1.numpy(),
            [[0, -1, 0, -1, -1],
             [0, 0, 0, -1, 1]],
        )
        np.testing.assert_equal(
            y_2.numpy(),
            [[1, -1, 0, -1, -1],
             [0, 1, 0, -1, 1]],
        )

if __name__ == '__main__':
    unittest.main()