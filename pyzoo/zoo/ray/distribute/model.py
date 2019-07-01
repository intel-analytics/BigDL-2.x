#
# Copyright 2018 Analytics Zoo Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import logging
import os
import tempfile
import time

import numpy as np
import ray
import tensorflow as tf

from zoo.ray.data.dataset import RayDataSet
from zoo.ray.distribute.gvhelper import GVHelper
from zoo.ray.distribute.ps import ShardedParameterServer
from zoo.ray.distribute.worker import ModelWorker
from zoo.ray.util import utils
from zoo.ray.util.utils import MKLSetting

logger = logging.getLogger(__name__)

class ModelAdapter(MKLSetting):

    def __init__(self, inputs,
                 outputs,
                 targets,
                 loss,
                 optimizer,
                 grad_vars,
                 cores=None):
        super().__init__(cores)
        self.inputs = inputs
        self.outputs = outputs
        self.targets = targets
        self.loss = loss
        self.optimizer = optimizer
        self.sess = tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(
            intra_op_parallelism_threads=self.get_cores(),
                inter_op_parallelism_threads=self.get_cores()))
        self.sess.run(tf.global_variables_initializer())
        self.grad_vars = grad_vars

        self.gv_helper = GVHelper(
            sess=self.sess, grad_vars=self.grad_vars)

    def execute(self, feed_dict_data):
        """
        :param inputs:
        :return: Returning (loss, non-flat-gradients)
        """
        loss_gradients = self.sess.run(
            [self.loss] + [grad[0] for grad in self.grad_vars],
            feed_dict=feed_dict_data)
        return loss_gradients

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
                # TODO: capture the end sequence only or change other wy to iterate the dataset
        except Exception as e:
            pass
        return result / count


class ModelLite(object):
    def __init__(self, model_bytes, model_fn):
        self.model_bytes = model_bytes
        self.model_fn = model_fn

    def to_adapter(self):
        if self.model_fn:
            input, output, target, loss, optimizer, grad_vars = self.extract_from_model_fn()
        else:
            input, output, target, loss, optimizer, grad_vars = \
                self.extract_from_keras_model()

        return ModelAdapter(inputs=utils.to_list(input),
                                    outputs=utils.to_list(output),
                                    targets=utils.to_list(target),
                                    loss=loss,
                                    optimizer=optimizer,
                                    grad_vars=grad_vars)

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

        import pickle
        return pickle.loads(model_bytes)

    @staticmethod
    def serialize_model(model):
        """Serialize model into byte array."""
        try:
            model_dir = tempfile.mkdtemp()
            model_path = os.path.join(model_dir, "model.h5")
            # TODO: change to definition only.
            model.save(str(model_path))
            with open(model_path, "rb") as file:
                return file.read()
        finally:
            # TODO: remove file here
            pass

    def extract_from_keras_model(self):
        """return input, output, target, loss, optimizer """
        import tensorflow.keras.backend as K
        try:
            keras_model = ModelLite.deserialize_model(self.model_bytes)
            loss = keras_model.total_loss
            inputs = keras_model.inputs
            outputs = keras_model.outputs
            targets = keras_model._targets #keras_model._targets
            vars = keras_model._collected_trainable_weights
            grads = K.gradients(loss, vars)
            optimizer = keras_model.optimizer
        except Exception as e:
            raise e
        return inputs, outputs, targets, loss, optimizer, list(zip(grads, vars))

class RayModel(object):
    """
    You should add your definition at model_fn
    and then return (input, output, target, loss, optimizer)
    """
    def __init__(self, model_bytes=None, model_fn=None):
        self.model_lite = ModelLite(model_bytes = model_bytes,
                                    model_fn = model_fn)

    @classmethod
    def from_model_fn(cls, model_fn):
        return cls(model_fn=model_fn)

    @classmethod
    def from_keras_model(cls, keras_model):
        model_bytes = ModelLite.serialize_model(keras_model)
        return cls(model_bytes = model_bytes)

    def fit(self, x, num_worker, batch_size, y=None, steps=10, strategy="ps"):
        self.batch_size = batch_size
        self.strategy=strategy
        self.num_worker = num_worker
        self.modelAdapter = self.model_lite.to_adapter()
        self.x = self._preprocess_input(x, y)
        self._init_distributed_engine()
        for i in range(1, steps + 1):
            self.step(i)
        self.modelAdapter.set_flat_parameters(ray.get(self.workers[0].get_weights.remote()))
        return self

    def _preprocess_input(self, x, y, repeat=True):
            #TODO: list of inputs
            if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
                return RayDataSet.from_dataset_generator(
                    input_fn=lambda:tf.data.Dataset.from_tensor_slices((x, y)),
                                                  repeat=True,
                                                  batch_size=int(self.batch_size / self.num_worker))
            elif isinstance(x, RayDataSet) and (y is None):
                return x
            else:
                raise TypeError("Unsupported training data type: %s" % type(x))

    def evaluate(self, x, y=None, metric_fn=None):
        ray_dataset = self._preprocess_input(x, y, repeat=False)
        # TODO: add metric_fn back
        return self.modelAdapter.evaluate(ray_dataset)

    def _init_distributed_engine(self):
        weights = self.modelAdapter.get_flat_parameters()
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
                                              modelLite=self.model_lite,
                                              ))

        logger.info(
            "Creating model workers ({} total)".format(
                self.num_worker))
        for worker_index in range(self.num_worker):
            self.workers.append(
                ModelWorker.remote(self.model_lite, self.x, self.num_worker))


    def step(self, step_id):
        start = time.time()
        # workers of sharded_grads
        sharded_grad_ids = []
        results = []
        losses = []
        for worker in self.workers:
            # 1) pull the latest weights from ps
            parameters = [ps.pull.remote() for ps in self.pss]
            # 2) compute the grads
            sharded_grad = worker.pull_and_execute._remote(args=parameters, kwargs=None, num_return_vals=self.num_worker)
            sharded_grad_ids.append(sharded_grad)
            losses.append(worker.get_loss.remote())

        grads_per_ps = list(zip(*sharded_grad_ids))
        assert len(grads_per_ps[0]) == self.num_worker, "we should get correct grads for each ps"
        # 3) push and aggregate grads on ps
        for index, grads in enumerate(grads_per_ps):
            results.append(self.pss[index].push.remote(*grads))
        # wait for complete
        ray.wait(object_ids=results, num_returns=len(results))
        end = time.time()
        avg_loss = np.mean([ray.get(loss) for loss in losses])
        throughput = self.batch_size * self.num_worker / (end - start)
        print("Iteration: {}, throughput: {}, loss: {}".format(step_id, throughput, avg_loss))


