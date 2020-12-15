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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either exp'
# ress or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from zoo.automl.search.ray_tune_search_engine import RayTuneSearchEngine
from zoo.automl.common.metrics import Evaluator
from zoo.automl.pipeline.time_sequence import TimeSequencePipeline
from zoo.automl.common.util import *
from zoo.automl.config.recipe import *
from zoo.ray import RayContext


ALLOWED_FIT_METRICS = ("mse", "mae", "r2")


class BasePredictor(object):

    def __init__(self,
                 name="automl",
                 logs_dir="~/zoo_automl_logs",
                 search_alg=None,
                 search_alg_params=None,
                 scheduler=None,
                 scheduler_params=None,
                 ):

        self.logs_dir = logs_dir
        self.name = name
        self.search_alg = search_alg
        self.search_alg_params = search_alg_params
        self.scheduler = scheduler
        self.scheduler_params = scheduler_params

    @abstractmethod
    def create_feature_transformer(self):
        raise NotImplementedError

    @abstractmethod
    def make_model_fn(self, resource_per_trail):
        raise NotImplementedError

    def _check_df(self, df):
        assert isinstance(df, pd.DataFrame) and df.empty is False, \
            "You should input a valid data frame"

    @staticmethod
    def _check_fit_metric(metric):
        if metric not in ALLOWED_FIT_METRICS:
            raise ValueError(f"metric {metric} is not supported for fit. "
                             f"Input metric should be among {ALLOWED_FIT_METRICS}")

    def fit(self,
            input_df,
            validation_df=None,
            metric="mse",
            recipe=SmokeRecipe(),
            mc=False,
            resources_per_trial={"cpu": 2},
            ):
        """
        Trains the model for time sequence prediction.
        If future sequence length > 1, use seq2seq model, else use vanilla LSTM model.
        :param input_df: The input time series data frame, Example:
         datetime   value   "extra feature 1"   "extra feature 2"
         2019-01-01 1.9 1   2
         2019-01-02 2.3 0   2
        :param validation_df: validation data
        :param metric: String. Metric used for train and validation. Available values are
                       "mean_squared_error" or "r_square"
        :param recipe: a Recipe object. Various recipes covers different search space and stopping
                      criteria. Default is SmokeRecipe().
        :param resources_per_trial: Machine resources to allocate per trial,
            e.g. ``{"cpu": 64, "gpu": 8}`
        :return: a pipeline constructed with the best model and configs.
        """
        self._check_df(input_df)
        if validation_df is not None:
            self._check_df(validation_df)

        ray_ctx = RayContext.get()
        is_local = ray_ctx.is_local
        # BasePredictor._check_fit_metric(metric)
        if not is_local:
            remote_dir = os.path.join(os.sep, "ray_results", self.name)
            if self.name not in get_remote_list(os.path.dirname(remote_dir)):
                cmd = "hadoop fs -mkdir -p {}".format(remote_dir)
                process(cmd)
        else:
            remote_dir = None

        self.pipeline = self._hp_search(
            input_df,
            validation_df=validation_df,
            metric=metric,
            recipe=recipe,
            mc=mc,
            resources_per_trial=resources_per_trial,
            remote_dir=remote_dir)
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
        :param metric: A list of Strings Available string values are "mean_squared_error",
                      "r_square".
        :return: a list of metric evaluation results.
        """
        Evaluator.check_metric(metric)
        return self.pipeline.evaluate(input_df, metric)

    def predict(self,
                input_df):
        """
        Predict future sequence from past sequence.
        :param input_df: The input time series data frame, Example:
         datetime   value   "extra feature 1"   "extra feature 2"
         2019-01-01 1.9 1   2
         2019-01-02 2.3 0   2
        :return: a data frame with 2 columns, the 1st is the datetime, which is the last datetime of
            the past sequence.
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
                   recipe,
                   mc,
                   resources_per_trial,
                   remote_dir):
        ft = self.create_feature_transformer()
        try:
            feature_list = ft.get_feature_list()
        except:
            feature_list = None

        model_fn = self.make_model_fn(resources_per_trial)

        # prepare parameters for search engine
        search_space = recipe.search_space(feature_list)

        searcher = RayTuneSearchEngine(logs_dir=self.logs_dir,
                                       resources_per_trial=resources_per_trial,
                                       name=self.name,
                                       remote_dir=remote_dir,
                                       )
        searcher.compile(input_df,
                         model_create_func=model_fn,
                         search_space=search_space,
                         recipe=recipe,
                         search_alg=self.search_alg,
                         search_alg_params=self.search_alg_params,
                         scheduler=self.scheduler,
                         scheduler_params=self.scheduler_params,
                         feature_transformers=ft,
                         validation_df=validation_df,
                         metric=metric,
                         mc=mc,
                         )
        # searcher.test_run()
        analysis = searcher.run()

        pipeline = self._make_pipeline(analysis,
                                       feature_transformers=ft,
                                       model=model_fn(),
                                       remote_dir=remote_dir)
        return pipeline

    def _print_config(self, best_config):
        print("The best configurations are:")
        for name, value in best_config.items():
            print(name, ":", value)

    def _make_pipeline(self, analysis, feature_transformers, model,
                       remote_dir):
        metric = "reward_metric"
        best_config = analysis.get_best_config(metric=metric, mode="max")
        best_logdir = analysis.get_best_logdir(metric=metric, mode="max")
        print("best log dir is ", best_logdir)
        dataframe = analysis.dataframe(metric=metric, mode="max")
        # print(dataframe)
        model_path = os.path.join(best_logdir, dataframe["checkpoint"].iloc[0])
        config = convert_bayes_configs(best_config).copy()
        self._print_config(config)
        if remote_dir is not None:
            all_config = restore_hdfs(model_path,
                                      remote_dir,
                                      feature_transformers,
                                      model,
                                      # config)
                                      )
        else:
            all_config = restore_zip(model_path,
                                     feature_transformers,
                                     model,
                                     # config)
                                     )
        return TimeSequencePipeline(name=self.name,
                                    feature_transformers=feature_transformers,
                                    model=model,
                                    config=all_config)
