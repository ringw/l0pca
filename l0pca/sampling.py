import numpy as np
import tensorflow as tf

def weighted_choice_naive(weights, shape):
    weights, _ = tf.linalg.normalize(tf.cast(weights, tf.float64), 1)
    def _weighted_impl(weights, shape):
        output = []
        values = np.arange(weights.shape[0])
        for i in range(shape[0]):
            output.append(np.random.choice(values, shape[1], replace=False, p=weights))
        return np.asarray(output)
    return tf.py_function(_weighted_impl, [weights, shape], tf.int32)

@tf.function
def weighted_choice(weights, shape):
    if len(shape) != 2:
        raise ValueError('Expected 2D (batches, samples)')
    n_elements = weights.shape[0]
    n_batches = shape[0]
    n_samples = shape[1]
    def random_pass_batch(size, batch):
        boolean_mask = tf.scatter_nd(
            batch[:, None],
            tf.ones([n_samples], tf.bool),
            [n_elements],
        )
        boolean_mask = tf.cond(
            (size > 0) & (batch[0] == 0),
            lambda: boolean_mask,
            lambda: tf.tensor_scatter_nd_update(
                    boolean_mask,
                    tf.constant([[0]]),
                    tf.constant([False]),
                ))
        weights_updated = tf.where(boolean_mask, 0., weights)
        weights_updated_scale = tf.linalg.norm(weights_updated, 1)
        cdf = tf.cumsum(weights_updated)
        samples = tf.random.uniform([
            n_samples - size]) * weights_updated_scale
        new_selected = tf.searchsorted(
            cdf, samples, 'right',
        )
        new_selected, _ = tf.unique(new_selected)
        new_selected = new_selected[:n_samples - size]
        orig_size = size
        size += tf.shape(new_selected)[0]
        batch = tf.concat(
            [
                batch[:orig_size],
                new_selected,
                tf.zeros([n_samples - size], tf.int32),
            ],
            axis=0,
        )
        batch = tf.sort(batch)
        return size, batch
    def random_pass_aug_batch(aug_batch):
        size = aug_batch[0]
        batch = aug_batch[1:]
        size, batch = random_pass_batch(size, batch)
        result = tf.concat([[size], batch], axis=0)
        return tf.ensure_shape(result, aug_batch.shape)
    initial_state = tf.zeros((n_batches, n_samples + 1), tf.int32)
    return tf.while_loop(
        lambda aug_batches: tf.reduce_any(
            aug_batches[:, 0] < n_samples),
        lambda aug_batches: [tf.map_fn(random_pass_aug_batch, aug_batches)],
        [initial_state],
    )[0][:, 1:]