import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from typing import Iterable

from seq2seq.munchhausen.MunchhausenFormater import MunchhausenFormatter


class MunchhausenOptimiser:
    def __init__(self, *optimizers):
        self._instance = tuple(optimizers)

    def get_optimizers(self) -> Iterable[tf.Tensor]:
        return self._instance

    def set_optimizers(self, *optimizers):
        self._instance = tuple(optimizers)


class MunchhausenTrainOptimiser(MunchhausenOptimiser):
    def __init__(
            self, munchhausen_net, q_loss: tf.Tensor, q_diff_loss: tf.Tensor,
            alpha1: float = 0.9,
            alpha2: float = 0.3,
            alpha3: float = 0.2,
            beta: float = 0.2,
            gamma: float = 0.2,
            sigma: float = 10,
            omega: float = 0.2,
            clip_norm: float = 0.03,
            MNS: int = 3
    ):
        self.D_optimizer = None  # type: tf.Tensor
        self.Q_optimizer = None  # type: tf.Tensor
        self.sample_diff_optimizer = None  # type: tf.Tensor
        with vs.variable_scope("train-optimizer"):
            optimizer = tf.train.AdamOptimizer()
            var_list = munchhausen_net.get_q_function_variables()
            self.D_optimizer = MunchhausenTrainOptimiser.clip_minimize(optimizer, q_diff_loss, clip_norm, var_list)
            optimizer = tf.train.MomentumOptimizer(0.003, 0.95, use_nesterov=True)
            var_list = munchhausen_net.get_analyser_variables()
            self.Q_optimizer = MunchhausenTrainOptimiser.clip_minimize(optimizer, q_loss, clip_norm, var_list)
        super().__init__(self.D_optimizer, self.Q_optimizer)
        self.E = None  # type: float
        self.dE = 0  # type: float
        self.D = None  # type: float
        self.dD = 0  # type: float
        self.Q = None  # type: float
        self.dQ = 0  # type: float
        self.NS = 0  # type: int
        self.MNS = MNS  # type: int
        self.alpha1 = alpha1  # type: float
        self.alpha2 = alpha2  # type: float
        self.alpha3 = alpha3  # type: float
        self.beta = beta  # type: float
        self.gamma = gamma  # type: float
        self.sigma = sigma  # type: float
        self.omega = omega  # type: float
        self.formatter = MunchhausenFormatter(
            heads=("optimizer", "E", "dE", "Q", "dQ", "D", "dD"),
            formats=("s",) + (".4f",) * 6,
            sizes=(20,) + (10,) * 6,
            rows=range(7),
            height=10
        )

    def update(self, Q: float, D: float, E: float):
        average_diff = lambda dV, pV, V, alpha: alpha * dV + (1 - alpha) * (V - (pV or V))
        self.E, self.dE = E, average_diff(self.dE, self.E, E, self.alpha1)
        self.D, self.dD = D, average_diff(self.dD, self.D, D, self.alpha2)
        self.Q, self.dQ = Q, average_diff(self.dQ, self.Q, Q, self.alpha3)
        if self.D_optimizer in self.get_optimizers():
            self.formatter.print("q-diff-optimizer", self.E, self.dE, self.Q, self.dQ, self.D, self.dD)
            if abs(self.dD) < self.beta:
                self.set_optimizers(self.Q_optimizer)
        elif self.Q_optimizer in self.get_optimizers():
            self.formatter.print("q-optimizer", self.E, self.dE, self.Q, self.dQ, self.D, self.dD)
            if self.dE > self.gamma:
                self.NS += 1
            if self.NS >= self.MNS or self.D > self.sigma:
                self.set_optimizers(self.D_optimizer)
                self.NS = 0

    @staticmethod
    def clip_minimize(optimizer, loss, clip_norm, var_list) -> tf.Tensor:
        grads_and_vars = optimizer.compute_gradients(loss, var_list)
        vars_with_grad = [v for g, v in grads_and_vars if g is not None]
        if not vars_with_grad:
            raise ValueError(
                "No gradients provided for any variable, check your graph for ops"
                " that do not support gradients, between variables %s and loss %s." %
                ([str(v) for _, v in grads_and_vars], loss))
        clipped_grads_and_vars = [(tf.clip_by_norm(grad, clip_norm), var) for grad, var in grads_and_vars]
        return optimizer.apply_gradients(clipped_grads_and_vars)


class MunchhausenPreTrainOptimiser(MunchhausenOptimiser):
    def __init__(self, munchhausen_net, q_diff_loss: tf.Tensor, sample_diff_loss: tf.Tensor):
        self.q_diff_optimizer = None  # type: tf.Tensor
        self.sample_diff_optimizer = None  # type: tf.Tensor
        with vs.variable_scope("pretrain-optimizer"):
            optimizer = tf.train.AdamOptimizer()
            var_list = munchhausen_net.get_q_function_variables()
            self.q_diff_optimizer = optimizer.minimize(q_diff_loss, var_list=var_list)
            var_list = munchhausen_net.get_analyser_variables()
            self.sample_diff_optimizer = optimizer.minimize(sample_diff_loss, var_list=var_list)
        super().__init__(self.q_diff_optimizer, self.sample_diff_optimizer)
