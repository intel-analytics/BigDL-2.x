import tensorflow as tf

from zoo.ray.allreduce.gvhelper import GVHelper
from zoo.ray.allreduce.ps import ShardedParameterServer
from zoo.ray.allreduce.sgd import ModelWorker
from zoo.ray.util import utils
import time
import ray
import logging
import io
import tempfile
import os
logger = logging.getLogger(__name__)

class TFModelLite(object):
    def __init__(self, model_bytes, model_fn):
        self.model_bytes = model_bytes
        self.model_fn = model_fn

    def resolve(self):
        rayModel = TFModelLite(self.model_bytes, self.model_fn)
        rayModel._action()
        return rayModel

    def _action(self):
        if self.model_fn:
            self.model_fn = self.model_fn
            input, output, target, loss, optimizer, grad_vars = self.extract_from_model_fn()
        else:
            self.model_bytes = self.model_bytes
            input, output, target, loss, optimizer, grad_vars = \
                TFModelLite.extract_from_keras_model(self.model_bytes)
        self.optimizer = optimizer
        self.inputs = utils.to_list(input)
        self.outputs = utils.to_list(output)
        self.targets = utils.to_list(target)
        self.loss = loss
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
            intra_op_parallelism_threads=22, inter_op_parallelism_threads=22))
        self.sess.run(tf.global_variables_initializer())
        self.grad_vars = grad_vars

        self.gv_helper = GVHelper(
            sess=self.sess, grad_vars=self.grad_vars)
        return self

    def extract_from_model_fn(self):
        input, output, target, loss, optimizer = self.model_fn()
        # A list of (gradient, variable) pairs
        grad_vars = [
            t for t in optimizer.compute_gradients(loss)
            if t[0] is not None
        ]
        return  input, output, target, loss, optimizer, grad_vars

    @staticmethod
    def deserialize_model(model_bytes):
        model_dir = tempfile.mkdtemp()
        model_path = os.path.join(model_dir, "model.h5")
        try:
            with open(model_path, "wb") as f:
                f.write(model_bytes)
        # TODO: remove file and add more exception handling
        except Exception as e:
            raise e
        loaded_model = tf.keras.models.load_model(model_path)
        return loaded_model

    @staticmethod
    def serialize_model(model):
        """Serialize model into byte array."""
        try:
            model_dir = tempfile.mkdtemp()
            model_path = os.path.join(model_dir, "model.h5")
            model.save(str(model_path))
            with open(model_path, "rb") as file:
                return file.read()
        finally:
            # TODO: remove file here
            pass


    @staticmethod
    def extract_from_keras_model(model_bytes):
        """return input, output, target, loss, optimizer """
        import tensorflow.keras.backend as K
        try:
            keras_model = TFModelLite.deserialize_model(model_bytes)
            loss = keras_model.total_loss
            inputs = keras_model.inputs
            outputs = keras_model.outputs
            targets = keras_model._targets
            vars = keras_model._collected_trainable_weights
            grads = K.gradients(loss, vars)
            optimizer = keras_model.optimizer
        except Exception as e:
            raise e
        return inputs, outputs, targets, loss, optimizer, list(zip(grads, vars))

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
        assert len(parameters.shape) == 1, \
            "we only accept 1D vector here, but got: {}".format(len(parameters.shape))
        self.gv_helper.set_flat(parameters)

        # The order is the same with optimizer.compute_gradient

    def get_flat_parameters(self):
        return self.gv_helper.get_flat()

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
        ray_dataset.action()
        try:
            while True:
                input_data, output_data = ray_dataset.next_batch()
                a = metric_fn(self.sess,
                          self.inputs,
                          self.outputs,
                          self.targets,
                          input_data, output_data)
                result = result + a
                count = count + 1
                print(count)
        except Exception:
            pass
        return result/count


class RayModel(object):
    """
    You should add your definition at model_fn
    and then return (input, output, target, loss, optimizer)
    """
    def __init__(self, model_bytes=None, model_fn=None):
        self.model_lite = TFModelLite(model_bytes = model_bytes,
                                      model_fn = model_fn)

    @classmethod
    def from_model_fn(cls, model_fn):
        return cls(model_fn=model_fn)

    @classmethod
    def from_keras_model(cls, keras_model):
        model_bytes = TFModelLite.serialize_model(keras_model)
        return cls(model_bytes = model_bytes)

    # we can support x, y later
    def fit(self, ray_dataset_train, num_worker=2, steps=10):
        self.ray_dataset_train = ray_dataset_train
        self.num_worker = num_worker
        self.batch_size = ray_dataset_train.get_batchsize()
        self.resolved_model = self.model_lite.resolve()
        self._init_distributed_engine()
        for i in range(1, steps + 1):
            self.step(i)
        self.resolved_model.set_flat_parameters(ray.get(self.workers[0].get_weights.remote()))
        return self

    def _init_distributed_engine(self):
        weights = self.resolved_model.get_flat_parameters()
        sharded_weights = utils.split(weights, self.num_worker)
        # This weights would be used for both PS and ModelWorker
        sharded_weight_ids = [ray.put(w) for w in sharded_weights]
        self.workers = []
        self.pss = []
        logger.info(
            "Creating parameter server ({} total)".format(
                self.num_worker))

        for ps_index in range(self.num_worker):
            self.pss.append(
                ShardedParameterServer.remote(sharded_weight_ids[ps_index],
                                              ray_model=self.model_lite))

        logger.info(
            "Creating model workers ({} total)".format(
                self.num_worker))
        for worker_index in range(self.num_worker):
            self.workers.append(
                ModelWorker.remote(self.model_lite, self.ray_dataset_train, self.num_worker))


    def step(self, step_id):
        start = time.time()
        # workers of sharded_grads
        sharded_grad_ids = []
        results = []
        for worker in self.workers:
            # 1) pull the latest weights from ps
            parameters = [ps.get_parameters.remote() for ps in self.pss]
            # 2) compute the grads
            sharded_grad = worker.set_parameters_compute_gradients._remote(args=parameters, kwargs=None, num_return_vals=self.num_worker)
            sharded_grad_ids.append(sharded_grad)

        # print("Iteration: {}, loss is {}".format(step_id, np.mean([ray.get(loss) for loss in losses])))

        grads_per_ps = list(zip(*sharded_grad_ids))
        assert len(grads_per_ps[0]) == self.num_worker, "we should get correct grads for each ps"
        # 3) push and aggregate grads on ps
        for index, grads in enumerate(grads_per_ps):
            results.append(self.pss[index].apply_gradients.remote(*grads))
        # wait for complete
        ray.wait(object_ids=results, num_returns=len(results))
        end = time.time()
        print("Iteration: {}, throughput: {}".format(step_id, self.batch_size * self.num_worker / (
        end - start)))


