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
import itertools

import numpy as np
from pyspark.context import SparkContext
from spark_runner import SparkRunner

from zoo.orca import OrcaContext

logger = logging.getLogger(__name__)

class Estimator(object):
    @staticmethod
    def from_keras(*,
                   model_creator,
                   config=None,
                   verbose=False,
                   compile_args_creator=None
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
        :param compile_args_creator: (dict -> dict of loss, optimizer and metrics) Only used when
               the backend="horovod". This function takes in the `config` dict and returns a
               dictionary like {"optimizer": tf.keras.optimizers.SGD(lr), "loss":
               "mean_squared_error", "metrics": ["mean_squared_error"]}
        """
        return SparkTFEstimator(model_creator=model_creator,
                                config=config, verbose=verbose, 
                                compile_args_creator=compile_args_creator)


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

class SparkTFEstimator(Estimator):
    def __init__(self,
                 model_creator,
                 config=None,
                 compile_args_creator=None,
                 verbose=False):
        self.model_creator = model_creator
        self.compile_args_creator = compile_args_creator
        self.config = {} if config is None else config
        self.verbose = verbose

        sc = OrcaContext.get_spark_context()
        self.worker_nums = sc.getConf().get("spark.executor.instances")
        self.part_nums = self.worker_nums * 2

        if "batch_size" in self.config:
            raise Exception("Please do not specify batch_size in config. Input batch_size in the"
                            " fit/evaluate function of the estimator instead.")
        sc.stop()
        
    
    def fit(self, data, validation_data, epochs=1, batch_size=32, verbose=1,
            partition_len=32, callbacks=None, class_weight=None):
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
        :return:
        """
        import numpy as np
        sc = OrcaContext.get_spark_context()

        params = dict(
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks,
            class_weight=class_weight
        )

        params["model_creator"] = self.model_creator
        params["data_creator"] = data
        params["validation_data_creator"] = validation_data

        worker_nums = self.worker_nums
        part_nums = self.part_nums
        workerRDD = sc.parallelize(list(range(partition_len)), part_nums).repartition(worker_nums)
        spark_func = SparkRunner(**params).step
        res = workerRDD.barrier().mapPartitions(spark_func).collect()

        sc.stop()
        return res

    def evaluate(self, data, validation_data, epochs=1, batch_size=32, verbose=1, 
                partition_len=32, callbacks=None, class_weight=None):
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
        import numpy as np
        sc = OrcaContext.get_spark_context()

        logger.info("Starting validation step.")

        params = dict(
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks,
            class_weight=class_weight
        )

        params["model_creator"] = self.model_creator
        params["data_creator"] = data
        params["validation_data_creator"] = validation_data

        worker_nums = self.worker_nums
        part_nums = self.part_nums
        workerRDD = sc.parallelize(list(range(partition_len)), part_nums).repartition(worker_nums)
        spark_func = SparkRunner(**params).validate
        res = workerRDD.barrier().mapPartitions(spark_func).collect()
        
        sc.stop()
        return res
    

