import tensorflow as tf

from zoo.ray.allreduce.gvhelper import GVHelper
from zoo.ray.util import utils


class RayModel(object):
    """
    You should add your definition at model_fn
    and then return (input, output, target, loss, optimizer)
    """
    def __init__(self, model_bytes=None, model_fn=None, resolved=False):
        self.model_bytes = model_bytes
        self.model_fn = model_fn

    def resolve(self):
        rayModel = RayModel(self.model_bytes, self.model_fn)
        rayModel._action()
        return rayModel

    def _action(self):
        if self.model_fn:
            self.model_fn = self.model_fn
            input, output, target, loss, optimizer = self.model_fn()
        else:
            self.model_bytes = self.model_bytes
            input, output, target, loss, optimizer = \
                RayModel.extract_from_keras_model(self.model_bytes)
        self.optimizer = optimizer
        self.inputs = utils.to_list(input)
        self.outputs = utils.to_list(output)
        self.targets = utils.to_list(target)
        self.loss = loss
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
   intra_op_parallelism_threads=22, inter_op_parallelism_threads=22))
        self.sess.run(tf.global_variables_initializer())

        # A list of (gradient, variable) pairs
        self.grad_vars = [
            t for t in optimizer.compute_gradients(self.loss)
            if t[0] is not None
        ]
        self.gv_helper = GVHelper(
            sess=self.sess, grad_vars=self.grad_vars)
        return self

    @staticmethod
    def extract_from_keras_model(kerasmodel):
        pass

    @classmethod
    def from_model_fn(cls, model_fn):
        return cls(model_fn=model_fn)

    @classmethod
    def from_keras_model(cls, keras_model):
        # keras_model.save(path)
        # bytearray = load from path()
        # cls()
        pass

    def calc_accuracy(sess, inputs_op, outputs_op, targets_op, input_data, output_data):
        with tf.name_scope('accuracy'):
            # label [-1, 1] not one-hot encoding. If the shape mismatch, the result would be incorrect
            # as `tf.equal` would broadcast automatically during the comparing stage.
            correct_prediction = tf.equal(tf.argmax(outputs_op[0], 1),
                                          tf.cast(tf.reshape(targets_op[0], (-1,)), tf.int64))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
            accuracy = tf.reduce_mean(correct_prediction)
            return sess.run(accuracy,
                            feed_dict={targets_op[0]: output_data, inputs_op[0]: input_data})

    def evaluate(self, ray_dataset, metric_fn=calc_accuracy):
        result = 0
        count = 0
        while ray_dataset.has_next():
            input_data, output_data = ray_dataset.next_batch()
            a = metric_fn(self.sess,
                      self.inputs,
                      self.outputs,
                      self.targets,
                      input_data, output_data)
            result = result + a
            count = count + 1
            print(count)
        return result/count



    def compute_gradients(self, feed_dict_data):
        """
        :param inputs:
        :return: The returning gradient is not a flat gradient and it's divided by vars
        """
        gradients = self.sess.run(
            [grad[0] for grad in self.grad_vars],
            feed_dict=feed_dict_data)
        loss = self.sess.run(
            self.loss,
            feed_dict=feed_dict_data)
        return gradients, loss


    def set_flat_parameters(self, parameters):
        """
        :param parameters: 1D vector
        :return:
        """
        assert len(parameters.shape) == 1,\
            "we only accept 1D vector here, but got: {}".format(len(parameters.shape))
        self.gv_helper.set_flat(parameters)

    # The order is the same with optimizer.compute_gradient
    def get_flat_parameters(self):
        return self.gv_helper.get_flat()

