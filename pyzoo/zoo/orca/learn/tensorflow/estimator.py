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
from bigdl.optim.optimizer import MaxIteration, SGD
from zoo.orca.data.shard import SparkXShards
from zoo.tfpark import TFEstimator, TFOptimizer, TFPredictor, TFNet
import pandas as pd
import tensorflow as tf

from zoo.tfpark.tf_dataset import TFDataset
from zoo.util import nest


class Estimator(object):
    def fit(self, data, steps, **kwargs):
        pass

    def predict(self, data, **kwargs):
        pass

    @staticmethod
    def from_graph(*, inputs, outputs=None,
                   labels=None, loss=None, optimizer=None,
                   metrics=None, updates=None,
                   sess=None, model_dir=None, backend="spark"):
        assert backend == "spark", "only spark backend is supported for now"
        if sess is None:
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())

        return TFOptimizerWrapper(inputs=inputs,
                                  outputs=outputs,
                                  labels=labels,
                                  loss=loss,
                                  optimizer=optimizer,
                                  metrics=metrics, updates=updates,
                                  sess=sess,
                                  model_dir=model_dir)


def _xshards_to_tf_dataset(data_shard,
                           batch_size=-1, batch_per_thread=-1,
                           validation_data_shard=None):
    # todo data_shard.head ?
    from zoo.orca.data.utils import get_spec, flatten_xy

    (feature_spec, label_spec) = data_shard.rdd.map(get_spec).first()

    feature_spec = [(tf.dtypes.as_dtype(spec[0]), spec[1]) for spec in feature_spec]
    label_spec = [(tf.dtypes.as_dtype(spec[0]), spec[1]) for spec in label_spec] \
        if label_spec is not None else None

    assert batch_size != -1 or batch_per_thread != -1, \
        "one of batch_size and batch_per_thread should be specified"

    val_rdd = None if validation_data_shard is None \
        else validation_data_shard.rdd.flatMap(flatten_xy)

    dataset = TFDataset.from_rdd(data_shard.rdd.flatMap(flatten_xy),
                                 features=feature_spec,
                                 labels=label_spec,
                                 batch_size=batch_size,
                                 batch_per_thread=batch_per_thread,
                                 val_rdd=val_rdd)

    return dataset


class TFOptimizerWrapper(Estimator):

    def __init__(self, *, inputs, outputs, labels, loss,
                 optimizer,
                 metrics,
                 updates, sess,
                 model_dir):
        self.inputs = inputs
        self.outputs = outputs
        self.labels = labels
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.updates = updates
        self.sess = sess
        self.model_dir = model_dir

    def fit(self, data_shard, steps,
            batch_size=32,
            validation_data_shard=None,
            feed_dict=None,
            session_config=None):
        import bigdl

        assert self.labels is not None, \
            "labels is None; it should not be None in training"
        assert self.loss is not None, \
            "loss is None; it should not be None in training"
        assert self.optimizer is not None, \
            "optimizer is None; it not None in training"
        assert isinstance(self.optimizer, bigdl.optim.optimizer.OptimMethod), \
            "optimizer is of type {}, ".format(type(self.optimizer)) + \
            "it should be an instance of bigdl.optim.optimizer.OptimMethod"

        dataset = _xshards_to_tf_dataset(data_shard,
                                         batch_size=batch_size,
                                         validation_data_shard=validation_data_shard)

        if feed_dict is not None:
            tensor_with_value = {key: (value, value) for key, value in feed_dict.items()}
        else:
            tensor_with_value = None

        optimizer = TFOptimizer.from_loss(
            loss=self.loss,
            optim_method=self.optimizer,
            inputs=(self.inputs, self.labels),
            dataset=dataset,
            metrics=self.metrics,
            updates=self.updates, session=self.sess,
            tensor_with_value=tensor_with_value,
            session_config=session_config,
            model_dir=self.model_dir)

        optimizer.optimize(end_trigger=MaxIteration(steps))
        return self

    def predict(self, data_shard, batch_size=32):
        assert self.outputs is not None, \
            "output is None, it should not be None in prediction"
        dataset = _xshards_to_tf_dataset(data_shard,
                                         batch_per_thread=batch_size)

        flat_inputs = nest.flatten(self.inputs)
        flat_outputs = nest.flatten(self.outputs)
        tfnet = TFNet.from_session(sess=self.sess, inputs=flat_inputs, outputs=flat_outputs)
        return tfnet.predict(dataset)
