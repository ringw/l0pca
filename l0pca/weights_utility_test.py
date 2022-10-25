import math
import numpy as np
import tensorflow as tf
import unittest
from l0pca import weights_utility

class WeightsUtilityTest(unittest.TestCase):

    def test_gradient_tape(self):
        t0 = 5.
        # L1 norm is equal to the temperature.
        param = tf.Variable(tf.fill([4], [t0 / 4]), name='param')

        @tf.function(jit_compile=True)
        def fake_model(param):
            loss = tf.constant(10.)
            update_ind = tf.constant([1])
            update_val = tf.constant([0.1])
            return loss, update_ind, update_val

        with tf.GradientTape() as tape:
            strategy = weights_utility.ParameterStrategy(t0)
            loss = strategy.call(param, fake_model)

        # Expected: Next step is increment of t0 / math.e.
        next_step_temperature = t0 * (1 + 1 / math.e)
        next_value = [1.25, 1.35, 1.25, 1.25]
        next_value = np.asarray(next_value) * next_step_temperature / np.linalg.norm(next_value, 1)
        # Gradient should point away from the desired destination.
        np.testing.assert_allclose(
            param.numpy() - next_value,
            tape.gradient(loss, param),
            atol=1e-5)

if __name__ == '__main__':
    unittest.main()