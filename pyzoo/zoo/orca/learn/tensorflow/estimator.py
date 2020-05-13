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
from zoo.orca.data.shard import SparkDataShards
from zoo.tfpark import TFEstimator
import pandas as pd
import numpy as np

from zoo.tfpark.tf_dataset import TFNdarrayDataset, TensorMeta


class Estimator(object):

    def fit(self, data, steps, batch_size):
        pass

    def predict(self, data, **kwargs):
        pass

    def evaluate(self, data, eval_methods, checkpoint_path):
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


def _data_shard_to_tf_dataset(data_shard, batch_size=-1, batch_per_thread=-1):

    first_data = data_shard.rdd.first()

    assert isinstance(first_data, pd.DataFrame)
    dtypes = list(first_data.dtypes)
    names = list(first_data.columns)

    assert batch_size != -1 or batch_per_thread != -1,\
        "one of batch_size and batch_per_thread should be specified"

    import tensorflow as tf
    tensor_structure = {n: TensorMeta(dtype=tf.dtypes.as_dtype(t),
                                      shape=(),
                                      name=n) for t, n in zip(dtypes, names)}

    def to_numpy(row):
        return {k: v for k, v in row.items()}

    def df_to_list(df):
        return [to_numpy(row) for _, row in df.iterrows()]

    return TFNdarrayDataset(data_shard.rdd.flatMap(df_to_list),
                            tensor_structure,
                            batch_size=batch_size,
                            batch_per_thread=batch_per_thread)


def _rdd_to_data_shard(rdd):
    rdd.mapPartitions()
    return SparkDataShards(rdd)


class TFEstimatorWrapper(Estimator):

    def __init__(self, tfpark_estimator):

        self.tfpark_estimator = tfpark_estimator

    def fit(self, data_shard, steps, batch_size=32):

        def input_fn():
            return _data_shard_to_tf_dataset(data_shard, batch_size)

        self.tfpark_estimator.train(input_fn, steps)
        return self

    def evaluate(self, data_shard, eval_methods, checkpoint_path=None, batch_size=32):
        def input_fn():
            return _data_shard_to_tf_dataset(data_shard,
                                             batch_per_thread=batch_size)
        return self.tfpark_estimator.evaluate(input_fn, eval_methods, checkpoint_path)

    def predict(self, data_shard, checkpoint_path=None, predict_keys=None):
        def input_fn():
            return _data_shard_to_tf_dataset(data_shard)
        predictions = self.tfpark_estimator.predict(input_fn,
                                                    predict_keys=predict_keys,
                                                    checkpoint_path=checkpoint_path)
        # todo predictions is rdd, maybe change to DataShard?
        return predictions
