"""Simulated annealing-controlled parameters in terms of TF SGD.

Thermodynamic temperature is reciprocal temperature. In a simulated annealing
schedule, thermodynamic temperature multiplier should be a scaled log over
time: `t0 * log(i + e)` (initial recip-temperature value is `t0`).

We will produce an annealing schedule of a categorical distribution. The
probabilities are parameterized as non-negative multinomial logits. We can push
the logits vector L1 norm to an expected magnitude (thermodynamic temperature)
which should increase logarithmically over time, so that dispersion increases
and thermodynamic entropy decreases over time.

We do not hold ourselves to linear steps; learning rate is variable. When we
apply an update (before multiplying by epsilon), then we should also apply the
first derivative of temperature/original L1 norm. This corresponds to stepwise
updates to both, but can be adjusted with small adjustable epsilon. We know the
original magnitude, so the magnitude update is calculated with a first-order
differential equation.
"""
import tensorflow as tf

class AnnealingDiffEq(object):
    def __init__(self, t0=1.):
        self.t0 = t0

    def temperature(self, i):
        return self.t0 / tf.math.log(i + 2)

    def compute_derivative(self, temperature):
        """Applies a first-order diff eq: `y' = f(y)`.

        The derivative `dt/di` (thermodynamic temperature - reciprocal) is:
        `t0 / (i + e)`. Then: `t' = t0 exp(-t / t0)`. All updates ("gradient")
        must be adjusted so that they are the difference between a new vector
        with the expected updated L1 norm, and the original vector. Otherwise,
        we would not step through annealing at the expected schedule.
        """
        return self.t0 * tf.math.exp(-temperature / self.t0)

class ParameterStrategy(object):
    def __init__(self) -> None:
        self.annealing_diff_eq = AnnealingDiffEq()

    def call(self, param, evaluation_fn):
        @tf.custom_gradient
        def custom_gradient_impl(param):
            def grad(upstream, variables=[]):
                unused_loss, update_ind, update_score = evaluation_fn(param)
                return (
                    upstream * self._update_scatter_backprop(param, update_ind, update_score),
                    [tf.zeros_like(var) for var in variables])

            return evaluation_fn(param)[0], grad

        return custom_gradient_impl(param)

    def _update_scatter_backprop(self, param, update_ind, update_score):
        @tf.function
        def update_impl(param, update_ind, update_score, annealing_diff_eq):
            temperature = tf.linalg.norm(param, 1)
            updated_temperature = temperature + annealing_diff_eq.compute_derivative(temperature)
            orig_param = param
            param = tf.tensor_scatter_nd_add(
                param,
                update_ind[:, None],
                update_score,
            )
            return -(param * updated_temperature / tf.linalg.norm(param, 1) - orig_param)
        return update_impl(param, update_ind, update_score, self.annealing_diff_eq)