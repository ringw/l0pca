from ast import Param
from l0pca import svd_metric
import tensorflow as tf
from l0pca import weights_utility

class L0PCALayer(tf.keras.layers.Layer):
    def __init__(self, k, ndim, param_formula) -> None:
        super().__init__()
        self.k = k
        self.ndim = ndim
        self.param_formula = param_formula

    def build(self, input_shape):
        self.feature_logits = self.add_weight(
            name='feature_logits',
            shape=[input_shape[1]],
            initializer='zero',
            trainable=True,
        )

    def call(self, cov_matrix):
        self.add_metric(
            name='greedy_feature_1_weight',
            value=tf.math.reduce_max(
                tf.nn.softmax(self.feature_logits)))
        greedy_feature = tf.argsort(-self.feature_logits)
        for i in range(min(5, self.k)):
            self.add_metric(name='greedy_feature_{}'.format(i + 1),
                value=tf.cast(greedy_feature[i], tf.float32))
        return weights_utility.ParameterStrategy().call(
            self.feature_logits,
            lambda param: svd_metric.evaluate_cov_eigh(
                cov_matrix, tf.nn.softmax(param),
                k=self.k, ndim=self.ndim, param_formula=self.param_formula))

class L0PCA(tf.keras.Model):

    def __init__(self, k=3, ndim=1, param_formula=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer = L0PCALayer(k, ndim, param_formula)

    def build(self, input_shape):
        self.cov_matrix = self.add_weight(
            name='cov_matrix',
            shape=[input_shape[1], input_shape[1]],
            initializer='zero',
            trainable=False,
        )

    def call(self, data):
        data = data - tf.math.reduce_mean(data, axis=0)[None, :]
        tf.summary.experimental.set_step(self._train_counter)
        def compute_cov_matrix():
            cov = tf.linalg.matmul(tf.transpose(data), data)
            cov /= tf.cast(tf.shape(data)[0], tf.float32) - tf.constant(1.)
            return cov
        cov_matrix = tf.cond(
            tf.equal(self._train_counter, 0),
            lambda: self.cov_matrix.assign(compute_cov_matrix()),
            lambda: self.cov_matrix)
        result = self.layer(cov_matrix)
        self.add_loss(result)
        return result