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


import numpy as np
import tempfile
import zipfile
import os
import shutil

from zoo.automl.search.abstract import *
from zoo.automl.search.RayTuneSearchEngine import RayTuneSearchEngine
from zoo.automl.common.metrics import Evaluator
from zoo.automl.feature.time_sequence import TimeSequenceFeatureTransformer

from zoo.automl.model import TimeSequenceModel
from zoo.automl.pipeline.time_sequence import TimeSequencePipeline, load_ts_pipeline
from zoo.automl.common.util import *
from abc import ABC, abstractmethod


class Recipe(ABC):

    @abstractmethod
    def search_space(self, all_available_features):
        pass

    @abstractmethod
    def runtime_params(self):
        pass


class BasicRecipe(Recipe):
    """
    A basic recipe which can be used to get a taste of how it works.
    tsp = TimeSequencePredictor(...,recipe = BasicRecipe(1))
    """

    def __init__(self, num_samples=1):
        self.num_samples = num_samples

    def search_space(self, all_available_features):
        return {
            # -------- feature related parameters
            "selected_features": RandomSample(
                lambda spec: np.random.choice(
                    all_available_features,
                    size=np.random.randint(low=3, high=len(all_available_features), size=1),
                    replace=False)),

            # --------- model related parameters
            "lr": 0.001,
            "lstm_1_units": GridSearch([16, 32]),
            "dropout_1": 0.2,
            "lstm_2_units": 8,
            "dropout_2": RandomSample(lambda spec: np.random.uniform(0.2, 0.5)),
            "batch_size": 1024,
        }

    def runtime_params(self):
        return {
            "training_iteration": 1,
            "num_samples": self.num_samples,
        }


class RandomRecipe(Recipe):
    """
    Pure random sample Recipe. Often used as baseline.
    tsp = TimeSequencePredictor(...,recipe = RandomRecipe(5))
    """

    def __init__(self, num_samples=5, reward_metric=-0.05):
        self.num_samples = num_samples
        self.reward_metric = reward_metric

    def search_space(self, all_available_features):
        return {
            # -------- feature related parameters
            "selected_features": RandomSample(
                lambda spec: np.random.choice(
                    all_available_features,
                    size=np.random.randint(low=3, high=len(all_available_features), size=1))
            ),

            # --------- model parameters
            "lstm_1_units": RandomSample(lambda spec: np.random.choice([8, 16, 32, 64, 128], size=1)[0]),
            "dropout_1": RandomSample(lambda spec: np.random.uniform(0.2, 0.5)),
            "lstm_2_units": RandomSample(lambda spec: np.random.choice([8, 16, 32, 64, 128], size=1)[0]),
            "dropout_2": RandomSample(lambda spec: np.random.uniform(0.2, 0.5)),

            # ----------- optimization parameters
            "lr": RandomSample(lambda spec: np.random.uniform(0.001, 0.01)),
            "batch_size": RandomSample(lambda spec: np.random.choice([32, 64, 1024], size=1, replace=False)[0]),
        }

    def runtime_params(self):
        return {
            "reward_metric": self.reward_metric,
            "training_iteration": 10,
            "num_samples": self.num_samples,
        }


class TimeSequencePredictor(object):
    """
    Trains a model that predicts future time sequence from past sequence.
    Past sequence should be > 1. Future sequence can be > 1.
    For example, predict the next 2 data points from past 5 data points.
    Output have only one target value (a scalar) for each data point in the sequence.
    Input can have more than one features (value plus several features)
    Example usage:
        tsp = TimeSequencePredictor()
        tsp.fit(input_df)
        result = tsp.predict(test_df)

    """
    def __init__(self,
                 name="automl",
                 logs_dir="~/zoo_automl_logs",
                 future_seq_len=1,
                 dt_col="datetime",
                 target_col="value",
                 extra_features_col=None,
                 drop_missing=True,
                 remote_root=None,
                 ray_ctx=None,
                 redis_address=None
                 ):
        """
        Constructor of Time Sequence Predictor
        :param logs_dir where the automl tune logs file located
        :param future_seq_len: the future sequence length to be predicted
        :param dt_col: the datetime index column
        :param target_col: the target col (to be predicted)
        :param extra_features_col: extra features
        :param drop_missing: whether to drop missing values in the input
        """
        self.logs_dir = logs_dir
        self.pipeline = None
        self.future_seq_len = future_seq_len
        self.dt_col = dt_col
        self.target_col = target_col
        self.extra_features_col = extra_features_col
        self.drop_missing = drop_missing
        self.name = name
        self.ray_ctx = ray_ctx
        self.redis_address = redis_address
        if remote_root is not None:
            self.remote_dir = os.path.join(remote_root, "ray_results", name)
            if name not in get_remote_list(os.path.join(remote_root, "ray_results")):
                cmd = "hadoop fs -mkdir -p {}".format(self.remote_dir)
                process(cmd)
        else:
            self.remote_dir = None

    def describe(self):
        print('Pipeline is initialized with:')
        print('log_dir:', self.logs_dir)
        print('future_seq_len:', self.future_seq_len)
        print('dt_col:', self.dt_col)
        print('target_col:', self.target_col)
        print('extra_feature_col:', self.extra_features_col)
        print('drop_missing:', self.drop_missing)

    def fit(self,
            input_df,
            validation_df=None,
            metric="mean_squared_error",
            recipe=BasicRecipe(1)):
        """
        Trains the model for time sequence prediction.
        If future sequence length > 1, use seq2seq model, else use vanilla LSTM model.
        :param input_df: The input time series data frame, Example:
         datetime   value   "extra feature 1"   "extra feature 2"
         2019-01-01 1.9 1   2
         2019-01-02 2.3 0   2
        :param validation_df: validation data
        :param metric: String. Metric used for train and validation. Available values are "mean_squared_error" or
        "r_square"
        :param recipe: a Recipe object. Various recipes covers different search space and stopping criteria.
        Default is BasicRecipe(1).
        :return: self
        """
        #check if cols are in the df
        cols_list = [self.dt_col,self.target_col]
        if self.extra_features_col is not None:
            if not isinstance(self.extra_features_col, (list,)):
                raise ValueError("extra_features_col needs to be either None or a list")
            cols_list.extend(self.extra_features_col)

        missing_cols = set(cols_list) - set(input_df.columns)
        if len(missing_cols) != 0:
            raise ValueError("Missing Columns in the input dataframe:"+ ','.join(list(missing_cols)))

        if not Evaluator.check_metric(metric):
            raise ValueError("metric" + metric + "is not supported")

        self.pipeline = self._hp_search(input_df,
                                        validation_df=validation_df,
                                        metric=metric,
                                        recipe=recipe)
        return self.pipeline

    def evaluate(self,
                 input_df,
                 metric=None
                 ):
        """
        Evaluate the model on a list of metrics.
        :param input_df: The input time series data frame, Example:
         datetime   value   "extra feature 1"   "extra feature 2"
         2019-01-01 1.9 1   2
         2019-01-02 2.3 0   2
        :param metric: A list of Strings Available string values are "mean_squared_error", "r_square".
        :return: a list of metric evaluation results.
        """
        if not Evaluator.check_metric(metric):
            raise ValueError("metric" + metric + "is not supported")
        return self.pipeline.evaluate(input_df, metric)

    def predict(self,
                input_df):
        """
        Predict future sequence from past sequence.
        :param input_df: The input time series data frame, Example:
         datetime   value   "extra feature 1"   "extra feature 2"
         2019-01-01 1.9 1   2
         2019-01-02 2.3 0   2
        :return: a data frame with 2 columns, the 1st is the datetime, which is the last datetime of the past sequence.
            values are the predicted future sequence values.
            Example :
            datetime    value_0     value_1   ...     value_2
            2019-01-03  2           3                   9
        """
        return self.pipeline.predict(input_df)

    def _hp_search(self,
                   input_df,
                   validation_df,
                   metric,
                   recipe):

        ft = TimeSequenceFeatureTransformer(self.future_seq_len, self.dt_col, self.target_col, self.extra_features_col,
                                            self.drop_missing)

        feature_list = ft.get_feature_list(input_df)

        # model = VanillaLSTM(check_optional_config=False)
        model = TimeSequenceModel(check_optional_config=False, future_seq_len=self.future_seq_len)

        # prepare parameters for search engine
        search_space = recipe.search_space(feature_list)
        runtime_params = recipe.runtime_params()
        num_samples = runtime_params['num_samples']
        stop = dict(runtime_params)
        del stop['num_samples']

        searcher = RayTuneSearchEngine(logs_dir=self.logs_dir,
                                       ray_num_cpus=6,
                                       resources_per_trial={"cpu": 2},
                                       name=self.name,
                                       remote_dir=self.remote_dir,
                                       ray_ctx=self.ray_ctx,
                                       redis_address=self.redis_address)
        searcher.compile(input_df,
                         search_space=search_space,
                         stop=stop,
                         # feature_transformers=TimeSequenceFeatures,
                         feature_transformers=ft,  # use dummy features for testing the rest
                         model=model,
                         validation_df=validation_df,
                         metric=metric,
                         num_samples=num_samples)
        # searcher.test_run()

        trials = searcher.run()
        best = searcher.get_best_trials(k=1)[0]  # get the best one trial, later could be n
        pipeline = self._make_pipeline(best,
                                       feature_transformers=ft,
                                       # feature_transformers=TimeSequenceFeatures(
                                       #     file_path='../../../../data/nyc_taxi_rolled_split.npz'),
                                       model=model)
        return pipeline

    def _print_config(self, best_config):
        print("The best configurations are:")
        for name, value in best_config.items():
            print(name, ":", value)

    def _make_pipeline(self, trial, feature_transformers, model):
        isinstance(trial, TrialOutput)
        # TODO we need to save fitted parameters (not in config, e.g. min max for scalers, model weights)
        # for both transformers and model
        # temp restore from two files

        self._print_config(trial.config)
        if self.remote_dir is not None:
            all_config = restore_hdfs(trial.model_path, self.remote_dir, feature_transformers, model, trial.config)
        else:
            all_config = restore_zip(trial.model_path, feature_transformers, model, trial.config)

        return TimeSequencePipeline(feature_transformers=feature_transformers, model=model, config=all_config)


if __name__ == "__main__":
    dataset_path = os.getenv("ANALYTICS_ZOO_HOME") + "/bin/data/NAB/nyc_taxi/nyc_taxi.csv"
    df = pd.read_csv(dataset_path)
    from zoo.automl.common.util import split_input_df

    train_df, val_df, test_df = split_input_df(df, val_split_ratio=0.1, test_split_ratio=0.1)

    # print(train_df.describe())
    # print(test_df.describe())
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--redis_address",
                        default=None,
                        type=str,
                        help="redis address of ray cluster")
    parser.add_argument("--hadoop_conf",
                        default=None,
                        type=str,
                        help="turn on yarn mode by passing the path to the hadoop configuration folder.")
    parser.add_argument("--spark_local",
                        default=False,
                        type=bool,
                        help="Run on spark local")
    args = parser.parse_args()

    if not args.hadoop_conf and not args.redis_address and not args.spark_local:
        print("*****Run automl on local*****")
        remote_root = None
        ray_ctx = None
    else:
        remote_root = "hdfs://172.16.0.103:9000"
        if args.hadoop_conf:
            print("*****Run automl with RayOnSpark*****")
            from zoo import init_spark_on_yarn
            from zoo.ray.util.raycontext import RayContext
            slave_num = 2
            sc = init_spark_on_yarn(
                hadoop_conf=args.hadoop_conf,
                conda_name="ray36",
                num_executor=slave_num,
                executor_cores=4,
                executor_memory="8g ",
                driver_memory="2g",
                driver_cores=4,
                extra_executor_memory_for_ray="10g")
            ray_ctx = RayContext(sc=sc, object_store_memory="5g")
        elif args.spark_local:
            from zoo import init_spark_on_local
            from zoo.ray.util.raycontext import RayContext
            print("*****Run automl on spark local*****")
            sc = init_spark_on_local(cores=4)
            ray_ctx = RayContext(sc=sc)
        else:
            print("*****Run automl with ray cluster. Redis_address is {}*****".format(args.redis_address))
            ray_ctx = None

    tsp = TimeSequencePredictor(dt_col="datetime",
                                target_col="value",
                                extra_features_col=None,
                                remote_root=remote_root,
                                ray_ctx=ray_ctx,
                                redis_address=args.redis_address
                                )
    # tsp.describe()
    pipeline = tsp.fit(train_df,
                       validation_df=val_df,
                       metric="mean_squared_error")

    print("evaluate:", pipeline.evaluate(test_df, metric=["mean_squared_error", "r_square"]))
    pred = pipeline.predict(test_df)
    print("predict:", pred.shape)

    save_pipeline_file = "tmp.ppl"
    pipeline.save(save_pipeline_file)
    tsp.describe()

    new_pipeline = load_ts_pipeline(save_pipeline_file)
    os.remove(save_pipeline_file)

    print("evaluate:", new_pipeline.evaluate(test_df, metric=["mean_squared_error", "r_square"]))

    new_pred = new_pipeline.predict(test_df)
    print("predict:", pred.shape)
    np.testing.assert_allclose(pred["value"].values, new_pred["value"].values)

    new_pipeline.fit(train_df, val_df, epoch_num=5)
    print("evaluate:", new_pipeline.evaluate(test_df, metric=["mean_squared_error", "r_square"]))


    # ray_ctx.stop()


