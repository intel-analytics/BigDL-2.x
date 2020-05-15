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
from bigdl.optim.optimizer import MaxIteration
from zoo.orca.data.shard import SparkDataShards
from zoo.tfpark import TFEstimator, TFOptimizer, TFPredictor, TFNet
import pandas as pd
import tensorflow as tf

from zoo.tfpark.tf_dataset import TFDataset
from zoo.util import nest


class Estimator(object):
    def fit(self, data, features, labels, steps, batch_size, **kwargs):
        pass

    def predict(self, data, features, **kwargs):
        pass

    @staticmethod
    def from_model_fn(model_fn,
                      model_dir=None,
                      config=None,
                      params=None,
                      warm_start_from=None, backend="spark"):
        assert backend == "spark", "only spark backend is supported for now"
        est = TFEstimator.from_model_fn(model_fn,
                                        model_dir=model_dir,
                                        config=config,
                                        params=params,
                                        warm_start_from=warm_start_from)
        return TFEstimatorWrapper(est)

    @staticmethod
    def from_pre_built_graph(*, inputs, outputs,
                             labels, loss, train_op,
                             metrics=None, updates=None,
                             sess=None, model_dir=None, backend="spark"):
        assert backend == "spark", "only spark backend is supported for now"
        if sess is None:
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())

        return TFOptimizerWrapper(inputs=inputs,
                                  outputs=outputs,
                                  labels=labels,
                                  loss=loss, train_op=train_op,
                                  metrics=metrics, updates=updates,
                                  sess=sess,
                                  model_dir=model_dir)


def _data_shard_to_tf_dataset(data_shard, feature_cols, label_cols,
                              batch_size=-1, batch_per_thread=-1,
                              validation_data_shard=None):
    first_data = data_shard.rdd.first()

    assert isinstance(first_data, pd.DataFrame)
    dtypes = list(first_data.dtypes)
    names = list(first_data.columns)

    name2idx = {name: idx for idx, name in enumerate(names)}

    for feat in feature_cols:
        if feat not in name2idx:
            raise ValueError("Could not find feature column {} in DataShard".format(feat))

    if label_cols is not None:
        for label in label_cols:
            if label not in name2idx:
                raise ValueError("Could not find label column {} in DataShard".format(label))

    assert batch_size != -1 or batch_per_thread != -1, \
        "one of batch_size and batch_per_thread should be specified"

    import tensorflow as tf
    feature_spec = {feat: (tf.dtypes.as_dtype(dtypes[name2idx[feat]]), ())
                    for feat in feature_cols}
    if label_cols is not None:
        label_spec = {label: (tf.dtypes.as_dtype(dtypes[name2idx[label]]), ())
                      for label in label_cols}
    else:
        label_spec = None

    def select(row):
        features = {feat: row[feat] for feat in feature_cols}
        if label_cols is not None:
            labels = {label: row[label] for label in label_cols}
            return (features, labels)
        else:
            return (features,)

    def df_to_list(df):
        return [select(row) for _, row in df.iterrows()]

    val_rdd = None if validation_data_shard is None \
        else validation_data_shard.rdd.flatMap(df_to_list)

    dataset = TFDataset.from_rdd(data_shard.rdd.flatMap(df_to_list),
                                 features=feature_spec,
                                 labels=label_spec,
                                 batch_size=batch_size,
                                 batch_per_thread=batch_per_thread,
                                 val_rdd=val_rdd)

    return dataset


def _rdd_to_data_shard(rdd):
    rdd.mapPartitions()
    return SparkDataShards(rdd)


class TFOptimizerWrapper(Estimator):

    def __init__(self, inputs, outputs, labels, loss,
                 train_op, metrics,
                 updates, sess,
                 model_dir):
        self.inputs = inputs
        self.outputs = outputs
        self.labels = labels
        self.loss = loss
        self.train_op = train_op
        self.metrics = metrics
        self.updates = updates
        self.sess = sess
        self.model_dir = model_dir

    def fit(self, data_shard, features, labels, steps,
            batch_size=32,
            validation_data_shard=None,
            feed_dict=None,
            session_config=None):

        # todo 1. input keys should match features
        # todo 2. labels keys should match labels

        dataset = _data_shard_to_tf_dataset(data_shard,
                                            feature_cols=features,
                                            label_cols=labels,
                                            batch_size=batch_size,
                                            validation_data_shard=validation_data_shard)

        if feed_dict is not None:
            tensor_with_value = {key: (value, value) for key, value in feed_dict.items()}
        else:
            tensor_with_value = None

        optimizer = TFOptimizer.from_train_op(
            train_op=self.train_op,
            inputs=self.inputs,
            labels=self.labels,
            loss=self.loss, metrics=self.metrics,
            updates=self.updates, sess=self.sess,
            dataset=dataset,
            tensor_with_value=tensor_with_value,
            session_config=session_config,
            model_dir=self.model_dir)

        optimizer.optimize(end_trigger=MaxIteration(steps))
        return self

    def predict(self, data_shard, features, batch_size=32):
        dataset = _data_shard_to_tf_dataset(data_shard,
                                            feature_cols=features,
                                            label_cols=None,
                                            batch_per_thread=batch_size)

        flat_inputs = nest.flatten(self.inputs)
        flat_outputs = nest.flatten(self.outputs)
        tfnet = TFNet.from_session(sess=self.sess, inputs=flat_inputs, outputs=flat_outputs)
        return tfnet.predict(dataset)


class TFEstimatorWrapper(Estimator):
    def __init__(self, tfpark_estimator):

        self.tfpark_estimator = tfpark_estimator

    def fit(self, data_shard, features, labels, steps, batch_size=32, **kwargs):

        def input_fn():
            return _data_shard_to_tf_dataset(data_shard,
                                             feature_cols=features,
                                             label_cols=labels,
                                             batch_size=batch_size)

        self.tfpark_estimator.train(input_fn, steps)
        return self

    def predict(self, data_shard, features, batch_size=32, checkpoint_path=None, predict_keys=None):
        def input_fn():
            return _data_shard_to_tf_dataset(data_shard,
                                             feature_cols=features,
                                             label_cols=None,
                                             batch_per_thread=batch_size)

        predictions = self.tfpark_estimator.predict(input_fn,
                                                    predict_keys=predict_keys,
                                                    checkpoint_path=checkpoint_path)
        # todo predictions is rdd, maybe change to DataShard?
        return predictions
