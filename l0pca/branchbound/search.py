import tensorflow as tf

import model
import node

CAPACITY = 100000

def value_to_priority(spca, value):
    # The upper bound value 
    max_priority = tf.cast(1 << 24, tf.float32)
    return tf.cast((1. - value / spca.frobenius_norm) * max_priority, tf.int64)

def new_search_queue(spca):
    root_node_y, root_node_branch_var = node.build_root(spca)
    queue = tf.queue.PriorityQueue(
        CAPACITY,
        [tf.int32, tf.int32],
        [tf.TensorShape([spca.n]), tf.TensorShape([])],
    )
    queue.enqueue([tf.constant(0, tf.int64), root_node_y, root_node_branch_var])
    return queue

@tf.function
def step(spca, queue, best_obj, best_y):
    if queue.size() < CAPACITY:
        _, node_y, branch_node = queue.dequeue()
        node_1, bounds_1, proj_1, node_2, bounds_2, proj_2, node_best, node_best_bound = node.process_node(spca, node_y, branch_node)
        # Non-terminal node to be added to the queue.
        if bounds_1[1] > best_obj and bounds_1[0] != bounds_1[1]:
            queue.enqueue([value_to_priority(spca, bounds_1[1]), node_1, proj_1])
        if bounds_2[1] > best_obj and bounds_2[0] != bounds_2[1]:
            queue.enqueue([value_to_priority(spca, bounds_2[1]), node_2, proj_2])
        if node_best_bound > best_obj:
            return node_best_bound, node_best
        else:
            return best_obj, best_y
    else:
        return best_obj, best_y
