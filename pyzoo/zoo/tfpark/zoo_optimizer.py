import tensorflow as tf
from bigdl.optim.optimizer import OptimMethod
from zoo.util.tf import process_grad


class FakeOptimMethod(OptimMethod):

    def __init__(self):
        super(FakeOptimMethod, self).__init__(None, "float")


class ZooOptimizer(tf.train.Optimizer):
    """An optimizer that wraps another tf.Optimizer, using an allreduce to
    combine gradient values before applying gradients to model weights."""

    def __init__(self, optimizer, name=None):
        if name is None:
            name = "Zoo{}".format(type(optimizer).__name__)
        super(ZooOptimizer, self).__init__(name=name, use_locking=False)

        self._optimizer = optimizer

    def compute_gradients(self, *args, **kwargs):
        """Compute gradients of all trainable variables.
        See Optimizer.compute_gradients() for more info.
        In DistributedOptimizer, compute_gradients() is overriden to also
        allreduce the gradients before returning them.
        """
        gradients = self._optimizer.compute_gradients(*args, **kwargs)
        results = []
        for grad_var in gradients:
            grad = grad_var[0]
            var = grad_var[1]
            grad = process_grad(grad)
            with tf.control_dependencies([var]):
                grad_i = tf.identity(grad, name="zoo_identity_op_for_grad")
            results.append((grad_i, var))
        return results

    def apply_gradients(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._optimizer.apply_gradients(*args, **kwargs)

    def get_slot(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._optimizer.get_slot(*args, **kwargs)

    def get_slot_names(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._optimizer.get_slot_names(*args, **kwargs)

    def variables(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._optimizer.variables(*args, **kwargs)
