import tensorflow as tf

import model
import node

CAPACITY = 100000

def value_to_priority(spca, value):
    # The upper bound value 
    max_priority = tf.cast(1 << 24, tf.float32)
    return tf.cast((1. - value / spca.frobenius_norm) * max_priority, tf.int64)

def new_search_queue(spca):
    root_node = node.build_root(spca)
    queue = tf.queue.PriorityQueue(
        CAPACITY,
        [model.DTYPE],
        [tf.TensorShape([spca.n, 2])],
    )
    queue.enqueue([tf.constant(0, tf.int64), root_node])
    return queue

def step(spca, queue, best_obj, best_y):
    if queue.size() < CAPACITY:
        _, node_data = queue.dequeue()
        pri_1, node_1, pri_2, node_2, new_best_obj, new_best_y = node.process_node(spca, node_data)
        if pri_1 > best_obj:
            queue.enqueue([value_to_priority(spca, pri_1), node_1])
        if pri_2 > best_obj:
            queue.enqueue([value_to_priority(spca, pri_2), node_2])
        if new_best_obj > best_obj:
            return queue, new_best_obj, new_best_y
        else:
            return queue, best_obj, best_y
