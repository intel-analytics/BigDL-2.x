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
from zoo.tfpark import TFEstimator, TFOptimizer, TFNet
import pandas as pd

from zoo.tfpark.tf_dataset import TFDataset


class Estimator(object):
    def fit(self, data, features, labels, steps, batch_size):
        pass

    def predict(self, data, features, **kwargs):
        pass

    def evaluate(self, data, features, labels, eval_methods, checkpoint_path):
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


def _data_shard_to_tf_dataset(data_shard, feature_cols, label_cols,
                              batch_size=-1, batch_per_thread=-1):
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

    dataset = TFDataset.from_rdd(data_shard.rdd.flatMap(df_to_list),
                                 features=feature_spec,
                                 labels=label_spec,
                                 batch_size=batch_size,
                                 batch_per_thread=batch_per_thread)

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
            feed_dict=None,
            session_config=None):

        # todo 1. input keys should match features
        # todo 2. labels keys should match labels

        dataset = _data_shard_to_tf_dataset(data_shard,
                                            feature_cols=features,
                                            label_cols=labels,
                                            batch_size=batch_size)

        if feed_dict is not None:
            tensor_with_value = {key: (value, value) for key, value in feed_dict.items()}
        else:
            tensor_with_value = None

        optimizer = TFOptimizer.from_train_op(
            train_op=self.train_op,
            loss=self.loss, metrics=self.metrics,
            updates=self.updates, sess=self.sess,
            dataset=dataset,
            tensor_with_value=tensor_with_value,
            session_config=session_config,
            model_dir=self.model_dir)

        optimizer.optimize(end_trigger=MaxIteration(steps))
        return self

    def evaluate(self, data_shard, features, labels,
                 eval_methods, batch_size=32, checkpoint_path=None,
                 feed_dict=None, session_config=None):
        # eval_methods
        # key: string
        # value: a tf scala tensor that belongs to the current graph


        # todo how to handle feed_dict
        TFNet.from_session(sess=self.sess, inputs=self.inputs + self.labels, )


class TFEstimatorWrapper(Estimator):
    def __init__(self, tfpark_estimator):

        self.tfpark_estimator = tfpark_estimator

    def fit(self, data_shard, features, labels, steps, batch_size=32):

        def input_fn():
            return _data_shard_to_tf_dataset(data_shard,
                                             feature_cols=features,
                                             label_cols=labels,
                                             batch_size=batch_size)

        self.tfpark_estimator.train(input_fn, steps)
        return self

    def evaluate(self, data_shard, features, labels,
                 eval_methods, batch_size=32, checkpoint_path=None):

        # eval_methods
        # key: string
        # value: a function that takes predictions and labels and return a scala tensor

        def input_fn():
            return _data_shard_to_tf_dataset(data_shard,
                                             feature_cols=features,
                                             label_cols=labels,
                                             batch_per_thread=batch_size)

        return self.tfpark_estimator.evaluate(input_fn, eval_methods, checkpoint_path)

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
