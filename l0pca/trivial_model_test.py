"""Trivial fake example for executing PCA model."""
import l0pca.model
import tensorflow as tf

data = tf.random.normal((10, 100))
m = l0pca.model.L0PCA()
opt = tf.keras.optimizers.SGD(1e-5)
m.compile(opt)
m.fit(data, epochs=10)
import pdb; pdb.set_trace()