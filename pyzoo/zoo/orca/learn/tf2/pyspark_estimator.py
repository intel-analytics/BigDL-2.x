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
import itertools
import logging
import pickle
import tensorflow as tf

import numpy as np
from pyspark.conf import SparkConf
from pyspark.context import SparkContext

from zoo.orca.learn.tf2.tf_runner import TFRunner
from zoo.orca.learn.spark_estimator import Estimator as OrcaSparkEstimator

from zoo.orca.learn.utils import maybe_dataframe_to_xshards, dataframe_to_xshards, \
    convert_predict_xshards_to_dataframe, update_predict_xshards, \
    process_xshards_of_pandas_dataframe
from zoo.orca.data.utils import process_spark_xshards

logger = logging.getLogger(__name__)


class Estimator(object):
    @staticmethod
    def from_keras(*,
                   model_creator,
                   config=None,
                   verbose=False,
                   workers_per_node=1,
                   compile_args_creator=None,
                   backend="tf2",
                   cpu_binding=True,
                   ):
        """
        Create an Estimator for tensorflow 2.
        :param model_creator: (dict -> Model) This function takes in the `config`
               dict and returns a compiled TF model.
        :param config: (dict) configuration passed to 'model_creator',
               'data_creator'. Also contains `fit_config`, which is passed
               into `model.fit(data, **fit_config)` and
               `evaluate_config` which is passed into `model.evaluate`.
        :param verbose: (bool) Prints output of one model if true.
        :param workers_per_node: (Int) worker number on each node. default: 1.
        :param compile_args_creator: (dict -> dict of loss, optimizer and metrics) Only used when
               the backend="horovod". This function takes in the `config` dict and returns a
               dictionary like {"optimizer": tf.keras.optimizers.SGD(lr), "loss":
               "mean_squared_error", "metrics": ["mean_squared_error"]}
        :param backend: (string) You can choose "horovod" or "tf2" as backend. Default: `tf2`.
        :param cpu_binding: (bool) Whether to binds threads to specific CPUs. Default: True
        """
        return TensorFlow2SparkEstimator(model_creator=model_creator, config=config,
                                    verbose=verbose, workers_per_node=workers_per_node,
                                    backend=backend, compile_args_creator=compile_args_creator,
                                    cpu_binding=cpu_binding)


def make_data_creator(refs):
    def data_creator(config, batch_size):
        return refs

    return data_creator


def data_length(data):
    x = data["x"]
    if isinstance(x, np.ndarray):
        return x.shape[0]
    else:
        return x[0].shape[0]

class TensorFlow2SparkEstimator(OrcaSparkEstimator):
    def __init__(self,
                 model_creator,
                 compile_args_creator=None,
                 config=None,
                 verbose=False):
        self.model_creator = model_creator
        self.compile_args_creator = compile_args_creator
        self.config = {} if config is None else config
        self.verbose = verbose

        sc = SparkContext()

        if "batch_size" in self.config:
            raise Exception("Please do not specify batch_size in config. Input batch_size in the"
                            " fit/evaluate function of the estimator instead.")

        if "inter_op_parallelism" not in self.config:
            self.config["inter_op_parallelism"] = 1

        executor_cores = sc.getConf().get("spark.executor.cores")
        num_executors = sc.getConf().get("spark.executor.instances")

        self.num_workers = num_executors * executor_cores

    def fit(self, data, epochs=1, batch_size=32, verbose=1,
            callbacks=None, validation_data=None, class_weight=None,
            steps_per_epoch=None, validation_steps=None, validation_freq=1,
            data_config=None, feature_cols=None,
            label_cols=None):
        """
        Train this tensorflow model with train data.
        :param data: train data. It can be XShards, Spark DataFrame or creator function which
               returns Iter or DataLoader.
               If data is XShards, each partition can be a Pandas DataFrame or a dictionary of
               {'x': feature, 'y': label}, where feature(label) is a numpy array or a tuple of
               numpy arrays.
        :param epochs: Number of epochs to train the model. Default: 1.
        :param batch_size: Batch size used for training. Default: 32.
        :param verbose: Prints output of one model if true.
        :param callbacks: List of Keras compatible callbacks to apply during training.
        :param validation_data: validation data. Validation data type should be the same
               as train data.
        :param class_weight: Optional dictionary mapping class indices (integers) to a weight
               (float) value, used for weighting the loss function. This can be useful to tell
               the model to "pay more attention" to samples from an under-represented class.
        :param steps_per_epoch: Total number of steps (batches of samples) before declaring one
               epoch finished and starting the next epoch. If `steps_pre_epoch` is `None`, the
               epoch will run until the input dataset is exhausted. When passing an infinitely
               repeating dataset, you must specify the `step_per_epoch` argument.
        :param validation_steps: Total number of steps (batches of samples) to draw before stopping
               when performing validation at the end of every epoch. Default: None.
        :param validation_freq: Only relevant if validation data is provided. Integer of
               `collections_abc.Container` instance (e.g. list, tuple, etc.). If an integer,
               specifies how many training epochs to run before a new validation run is performed,
               e.g. `validation_freq=2` runs validation every 2 epochs. If a Container, specifies
               the epochs on which to run validation, e.g. `validation_freq=[1, 2, 10]` runs
               validation at the end of the 1st, 2nd, and 10th epochs.
        :param data_config: An optional dictionary that can be passed to data creator function.
        :param feature_cols: Feature column name(s) of data. Only used when data is a Spark
               DataFrame or an XShards of Pandas DataFrame. Default: None.
        :param label_cols: Label column name(s) of data. Only used when data is a Spark DataFrame or
               an XShards of Pandas DataFrame.
               Default: None.
        :return:
        """
        params = dict(
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks,
            class_weight=class_weight,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            validation_freq=validation_freq,
            data_config=data_config
        )

        sc = SparkContext()
        spark_rdd = sc.parallelize([1, 2, 3, 4], self.num_workers)

        from zoo.orca.data import SparkXShards
        data, validation_data = maybe_dataframe_to_xshards(data, validation_data,
                                                           feature_cols, label_cols,
                                                           mode="fit",
                                                           num_workers=self.num_workers,
                                                           accept_str_col=True)

        if isinstance(data, SparkXShards):
            if data._get_class_name() == 'pandas.core.frame.DataFrame':
                data, validation_data = process_xshards_of_pandas_dataframe(data, feature_cols,
                                                                            label_cols,
                                                                            validation_data, "fit")

            if validation_data is None:
                def transform_func(partition_refs):
                    params["data_creator"] = make_data_creator(partition_refs)
                    return TFRunner.step(**params)
                worker_stats = spark_rdd.mapPartitions(transform_func).collect()
            else:
                def zip_transform_func(this_partition_refs, that_partition_refs):
                    params["data_creator"] = make_data_creator(this_partition_refs)
                    params["validation_data_creator"] = make_data_creator(that_partition_refs)
                    return TFRunner.step(**params)
                worker_stats = spark_rdd.mapPartitions(zip_transform_func).collect()
        else:
            params["data_creator"] = data
            params["validation_data_creator"] = validation_data
            def train_func(*args):
                return TFRunner.step(**params)
            worker_stats = spark_rdd.mapPartitions(train_func).collect()

            worker_stats = list(itertools.chain.from_iterable(worker_stats))
        stats = worker_stats[0].copy()
        return stats


    def evaluate(self, data, batch_size=32, num_steps=None, verbose=1,
                 sample_weight=None, callbacks=None, data_config=None,
                 feature_cols=None, label_cols=None):
        """
        Evaluates the model on the validation data set.
        :param data: evaluate data. It can be XShards, Spark DataFrame or creator function which
               returns Iter or DataLoader.
               If data is XShards, each partition can be a Pandas DataFrame or a dictionary of
               {'x': feature, 'y': label}, where feature(label) is a numpy array or a tuple of
               numpy arrays.
        :param batch_size: Batch size used for evaluation. Default: 32.
        :param num_steps: Total number of steps (batches of samples) before declaring the evaluation
               round finished. Ignored with the default value of `None`.
        :param verbose: Prints output of one model if true.
        :param sample_weight: Optional Numpy array of weights for the training samples, used for
               weighting the loss function. You can either pass a flat (1D) Numpy array with the
               same length as the input samples (1:1 mapping between weights and samples), or in
               the case of temporal data, you can pass a 2D array with shape (samples,
               sequence_length), to apply a different weight to every timestep of every sample.
        :param callbacks: List of Keras compatible callbacks to apply during evaluation.
        :param data_config: An optional dictionary that can be passed to data creator function.
        :param feature_cols: Feature column name(s) of data. Only used when data is a Spark
               DataFrame or an XShards of Pandas DataFrame. Default: None.
        :param label_cols: Label column name(s) of data. Only used when data is a Spark DataFrame or
               an XShards of Pandas DataFrame.
               Default: None.
        :return: validation result
        """
        logger.info("Starting validation step.")
        params = dict(
            batch_size=batch_size,
            verbose=verbose,
            sample_weight=sample_weight,
            steps=num_steps,
            callbacks=callbacks,
            data_config=data_config,
        )
        from zoo.orca.data import SparkXShards

        sc = SparkContext()
        spark_rdd = sc.parallelize([1, 2, 3, 4], self.num_workers)

        data, _ = maybe_dataframe_to_xshards(data,
                                             validation_data=None,
                                             feature_cols=feature_cols,
                                             label_cols=label_cols,
                                             mode="evaluate",
                                             num_workers=self.num_workers,
                                             accept_str_col=True)

        if isinstance(data, SparkXShards):
            if data._get_class_name() == 'pandas.core.frame.DataFrame':
                data = process_xshards_of_pandas_dataframe(data, feature_cols, label_cols)

            data = data
            if data.num_partitions() != self.num_workers:
                data = data.repartition(self.num_workers)

            def transform_func(partition_refs):
                params["data_creator"] = make_data_creator(partition_refs)
                return TFRunner.validate(**params)

            worker_stats = spark_rdd.mapPartitions(transform_func).collect()
        else:  # data_creator functions; should return Iter or DataLoader
            params["data_creator"] = data
            def train_func(*args):
                return TFRunner.step(**params) 
            worker_stats = spark_rdd.mapPartitions(train_func).collect()

            worker_stats = list(itertools.chain.from_iterable(worker_stats))
        stats = worker_stats[0].copy()
        return stats
    
