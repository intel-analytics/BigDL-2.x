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
from pyspark.sql.dataframe import DataFrame
import numpy as np
from zoo.orca.learn.tf2.spark_runner import SparkRunner
from zoo.orca.learn.tf2.estimator import Estimator
from zoo.orca.learn.utils import maybe_dataframe_to_xshards, dataframe_to_xshards, \
    convert_predict_xshards_to_dataframe, update_predict_xshards, \
    process_xshards_of_pandas_dataframe
from zoo.orca.data.shard import SparkXShards
from zoo.orca.data.utils import ray_partition_get_data_label, _convert_list_tuple, combine

from zoo.orca import OrcaContext

logger = logging.getLogger(__name__)

# class Estimator(object):
#     @staticmethod
#     def from_keras(*,
#                    model_creator,
#                    config=None,
#                    verbose=False,
#                    compile_args_creator=None
#                    ):
#         """
#         Create an Estimator for tensorflow 2.
#         :param model_creator: (dict -> Model) This function takes in the `config`
#                dict and returns a compiled TF model.
#         :param config: (dict) configuration passed to 'model_creator',
#                'data_creator'. Also contains `fit_config`, which is passed
#                into `model.fit(data, **fit_config)` and
#                `evaluate_config` which is passed into `model.evaluate`.
#         :param verbose: (bool) Prints output of one model if true.
#         :param compile_args_creator: (dict -> dict of loss, optimizer and metrics) Only used when
#                the backend="horovod". This function takes in the `config` dict and returns a
#                dictionary like {"optimizer": tf.keras.optimizers.SGD(lr), "loss":
#                "mean_squared_error", "metrics": ["mean_squared_error"]}
#         """
#         return SparkTFEstimator(model_creator=model_creator,
#                                 config=config, verbose=verbose,
#                                 compile_args_creator=compile_args_creator)
#
#
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
        self.worker_nums = int(sc.getConf().get("spark.executor.instances"))
        self.part_nums = self.worker_nums * 2
        self.model_weights = None

        if "batch_size" in self.config:
            raise Exception("Please do not specify batch_size in config. Input batch_size in the"
                            " fit/evaluate function of the estimator instead.")
        
    
    def fit(self, data, epochs=1, batch_size=32, verbose=1,
            callbacks=None, validation_data=None, class_weight=None,
            steps_per_epoch=None, validation_steps=None, validation_freq=1,
            data_config=None, feature_cols=None,
            label_cols=None, model_dir=None):
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



        init_params = dict(
            model_creator=self.model_creator,
            compile_args_creator=self.compile_args_creator,
            config=self.config,
            verbose=self.verbose,
            size=self.worker_nums
        )

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

        # dataframe change to xshard, num_partition >= num_workers
        data, validation_data = maybe_dataframe_to_xshards(data, validation_data,
                                                           feature_cols, label_cols,
                                                           mode="fit",
                                                           num_workers=self.num_workers,
                                                           accept_str_col=True)
        if isinstance(data, SparkXShards):
            def combine_in_partition(partition_data):
                data_list = [data['x'] for data in partition_data]
                label_list = [data['y'] for data in partition_data]
                data = combine(data_list)
                label = combine(label_list)
                return {'x': data, 'y': label}

            # set train/validation data
            if validation_data is None:
                def transform_func(iter, init_param, param):
                    partition_data = iter.tolist()
                    # res = combine_in_partition(partition_data)
                    param["data_creator"] = make_data_creator(partition_data)
                    return SparkRunner(**init_param).step(**param)
                res = data.rdd.repartition(self.worker_nums).barrier()\
                    .mapPartitions(lambda iter: transform_func(iter, init_params, params)).collect()
            else:
                def transform_func(iter, init_param, param):
                    data_tuple_list = iter.tolist()
                    data_list = [x[0] for x in data_tuple_list]
                    valid_list = [x[1] for x in data_tuple_list]
                    # data = combine_in_partition(data_list)
                    # valid = combine_in_partition(valid_list)
                    param["data_creator"] = make_data_creator(data_list)
                    param["validation_data_creator"] = make_data_creator(valid_list)
                    return SparkRunner(**init_param).step(**param)

                res = data.zip(validation_data).rdd.repartition(self.worker_nums).barrier()\
                    .mapPartitions(lambda iter: transform_func(iter, init_params, params)).collect()
        else:
            params["data_creator"] = data
            params["validation_data_creator"] = validation_data

            workerRDD = sc.parallelize(list(range(self.worker_nums)), self.worker_nums).\
                repartition(self.worker_nums)

            def transform_func(iter, init_param, param):
                return SparkRunner(**init_param).step(**param)

            res = workerRDD.barrier().mapPartitions(lambda iter: transform_func(iter, init_params, params)).collect()

        self.model_weights = res[1]

        return res

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
        :param validation_data: validation data. Validation data type should be the same
               as train data.
        :param batch_size: Batch size used for evaluation. Default: 32.
        :param verbose: Prints output of one model if true.
        :param callbacks: List of Keras compatible callbacks to apply during evaluation.
        :param class_weight: Optional dictionary mapping class indices (integers) to a weight
               (float) value, used for weighting the loss function. This can be useful to tell
               the model to "pay more attention" to samples from an under-represented class.
        :return: validation result
        """
        import numpy as np
        sc = OrcaContext.get_spark_context()
        logger.info("Starting validation step.")

        init_params = dict(
            model_creator=self.model_creator,
            compile_args_creator=self.compile_args_creator,
            config=self.config,
            verbose=self.verbose,
            size=self.worker_nums,
            model_weights=self.model_weights
        )


        params = dict(
            batch_size=batch_size,
            verbose=verbose,
            sample_weight=sample_weight,
            steps=num_steps,
            callbacks=callbacks,
            data_config=data_config,
        )

        # dataframe change to xshard, num_partition >= num_workers
        data, _ = maybe_dataframe_to_xshards(data, validation_data=None,
                                             feature_cols=feature_cols,
                                             label_cols=label_cols,
                                             mode="evaluate",
                                             num_workers=self.num_workers,
                                             accept_str_col=True)
        if isinstance(data, SparkXShards):
            def combine_in_partition(partition_data):
                data_list = [data['x'] for data in partition_data]
                label_list = [data['y'] for data in partition_data]
                data = combine(data_list)
                label = combine(label_list)
                return {'x': data, 'y': label}

            # set train/validation data
            def transform_func(iter, init_param, param):
                partition_data = iter.tolist()
                # res = combine_in_partition(partition_data)
                param["data_creator"] = make_data_creator(partition_data)
                # param["model_weights"] = self.model_weights
                return SparkRunner(**init_param).validate(**param)

            res = data.rdd.repartition(self.worker_nums).barrier() \
                    .mapPartitions(lambda iter: transform_func(iter, init_params, params)).collect()
        else:
            params["data_creator"] = data
            # params["model_weights"] = self.model_weights

            worker_nums = self.worker_nums
            workerRDD = sc.parallelize(list(range(self.worker_nums)), worker_nums).repartition(worker_nums)
            def transform_func(iter, init_param, param):
                return SparkRunner(**init_param).validate(**param)
            res = workerRDD.barrier().mapPartitions(lambda iter: transform_func(iter, init_params, params)).collect()
        
        return res


    def predict(self, data, batch_size=None, verbose=1,
                steps=None, callbacks=None, data_config=None,
                feature_cols=None):
        """
        Predict the input data
        :param data: predict input data.  It can be XShards or Spark DataFrame.
               If data is XShards, each partition can be a Pandas DataFrame or a dictionary of
               {'x': feature}, where feature is a numpy array or a tuple of numpy arrays.
        :param batch_size: Batch size used for inference. Default: None.
        :param verbose: Prints output of one model if true.
        :param steps: Total number of steps (batches of samples) before declaring the prediction
               round finished. Ignored with the default value of None.
        :param callbacks: List of Keras compatible callbacks to apply during prediction.
        :param data_config: An optional dictionary that can be passed to data creator function.
        :param feature_cols: Feature column name(s) of data. Only used when data is a Spark
               DataFrame or an XShards of Pandas DataFrame. Default: None.
        :return:
        """
        logger.info("Starting predict step.")

        init_params = dict(
            model_creator=self.model_creator,
            compile_args_creator=self.compile_args_creator,
            config=self.config,
            verbose=self.verbose,
            size=self.worker_nums,
            model_weights=self.model_weights
        )

        params = dict(
            verbose=verbose,
            batch_size=batch_size,
            steps=steps,
            callbacks=callbacks,
            data_config=data_config
        )
        from zoo.orca.data import SparkXShards
        from pyspark.sql import DataFrame

        if isinstance(data, DataFrame):
            data = data.repartition(self.worker_nums)
            xshards, _ = dataframe_to_xshards(data,
                                              validation_data=None,
                                              feature_cols=feature_cols,
                                              label_cols=None,
                                              mode="predict",
                                              accept_str_col=True)
            def transform_func(iter, init_param, param):
                partition_data = iter.tolist()
                # res = combine_in_partition(partition_data)
                param["data_creator"] = make_data_creator(partition_data)
                return SparkRunner(**init_param).predict(**param)

            pred_shards = SparkXShards(xshards.rdd.repartition(self.worker_nums).barrier() \
                .mapPartitions(lambda iter: transform_func(iter, init_params, params)))
            result = convert_predict_xshards_to_dataframe(data, pred_shards)
        else:
            raise ValueError("Only xshards or Spark DataFrame is supported for predict")

        return result

