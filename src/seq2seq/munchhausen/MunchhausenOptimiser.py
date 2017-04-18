import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.training.optimizer import Optimizer as tf_Optimizer


class Optimizer:
    def __init__(self, name: str, loss_name: str, *optimisers):
        self.loss_name = loss_name  # type: str
        self.name = name  # type: str
        self.instances = tuple(optimisers)


class MunchhausenOptimiser:
    def __init__(self, optimizer: Optimizer):
        self.instance = optimizer  # type: Optimizer

    def get_name(self) -> str:
        return self.instance.name

    def get_loss_name(self) -> str:
        return self.instance.loss_name

    def get_tf_optimizers(self) -> tf_Optimizer:
        return self.instance.instances


class MunchhausenTrainOptimiser(MunchhausenOptimiser):
    DOWN_DIFF_LOSS = 10
    UP_DIFF_LOSS = 30

    def __init__(self, munchhausen_net, q_loss: tf.Tensor, q_diff_loss: tf.Tensor):
        self.q_diff_optimizer = None  # type: tf_Optimizer
        self.q_optimizer = None  # type: tf_Optimizer
        self.sample_diff_optimizer = None  # type: tf_Optimizer
        with vs.variable_scope("train-optimizer"):
            optimizer = tf.train.AdadeltaOptimizer(1e-3, 0.95)
            q_diff = optimizer.minimize(q_diff_loss, var_list=munchhausen_net.get_q_function_variables())
            optimizer = optimizer.minimize(q_loss, var_list=munchhausen_net.get_analyser_variables())
            self.q_diff_optimizer = Optimizer("adadelta", "q-diff", q_diff)
            self.q_optimizer = Optimizer("adadelta", "q", optimizer)
        super().__init__(self.q_diff_optimizer)

    def update(self, q_diff_loss: float):
        if self.instance == self.q_diff_optimizer:
            if q_diff_loss < MunchhausenTrainOptimiser.DOWN_DIFF_LOSS:
                self.instance = self.q_optimizer
        elif self.instance == self.q_optimizer:
            if q_diff_loss > MunchhausenTrainOptimiser.UP_DIFF_LOSS:
                self.instance = self.q_diff_optimizer


class MunchhausenPreTrainOptimiser(MunchhausenOptimiser):
    def __init__(self, munchhausen_net, q_loss: tf.Tensor, q_diff_loss: tf.Tensor, sample_diff_loss: tf.Tensor):
        self.q_diff_optimizer = None  # type: tf_Optimizer
        self.q_optimizer = None  # type: tf_Optimizer
        self.sample_diff_optimizer = None  # type: tf_Optimizer
        with vs.variable_scope("pretrain-optimizer"):
            optimizer = tf.train.AdadeltaOptimizer()
            # q = optimizer.minimize(q_loss, var_list=munchhausen_net.get_analyser_variables())
            q_diff = optimizer.minimize(q_diff_loss, var_list=munchhausen_net.get_q_function_variables())
            sample_diff = optimizer.minimize(sample_diff_loss, var_list=munchhausen_net.get_analyser_variables())
            self.q_diff_optimizer = Optimizer("adadelta", "diff", q_diff, sample_diff)
        super().__init__(self.q_diff_optimizer)
