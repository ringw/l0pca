from ast import Param
import tensorflow as tf
from l0pca import weights_utility

class L0PCALayer(tf.keras.layers.Layer):
    def __init__(self, t0=1) -> None:
        super().__init__()
        self.t0 = 1

    def build(self, input_shape):
        self.feature_logits = self.add_weight(
            shape=[input_shape[1]],
            initializer=lambda shape, dtype: tf.fill(shape, tf.constant(1. / self.t0, dtype)),
            trainable=True,
        )
        self.cov_matrix = self.add_weight(
            shape=[input_shape[1], input_shape[1]],
            initializer='zero',
            trainable=False,
        )
        self.epoch = self.add_weight(
            shape=[],
            initializer='zero',
            dtype=tf.int32,
            trainable=False,
        )

    def call(self, data):
        data = data - tf.math.reduce_mean(data, axis=0)[None, :]
        cov_matrix = tf.cond(
            tf.equal(self.epoch, 0),
            lambda: self.cov_matrix.assign(tf.linalg.matmul(tf.transpose(data), data)),
            lambda: self.cov_matrix)
        self.epoch.assign_add(1)
        return weights_utility.ParameterStrategy(self.t0).call(
            self.feature_logits,
            lambda param: (tf.constant(0.5), tf.constant([1]), tf.constant([0.1])))

class L0PCA(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer = L0PCALayer()

    def call(self, data):
        result = self.layer(data)
        self.add_loss(result)
        return result