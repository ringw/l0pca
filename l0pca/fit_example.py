import datetime
from l0pca import model
import numpy as np
import os.path
import tensorflow as tf

# tf.config.optimizer.set_jit(True)

m = model.L0PCA(k=3, ndim=1, param_formula='bernoulli')
opt = tf.keras.optimizers.Adagrad(10)
# opt = tf.keras.optimizers.SGD(5)
m.compile(opt)

data = np.loadtxt('Buettner2015Features@1024.csv', delimiter=',', dtype='object')
gene_names = data[0, 1:]
data = data[1:, 1:].astype(np.float32)
data = tf.constant(data)
# cov = tf.linalg.matmul(tf.transpose(data), data) / (data.shape[0] - 1)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
file_writer.set_as_default()
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(log_dir, 'model'),
    save_freq=100,
    save_weights_only=True,
)
m.fit(data, batch_size=data.shape[0], epochs=1500, callbacks=[tensorboard_callback, checkpoint_callback])
