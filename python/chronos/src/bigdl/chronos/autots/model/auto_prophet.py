# +
#
# Copyright 2016 The BigDL Authors.
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

import pandas as pd

from bigdl.orca.automl.auto_estimator import AutoEstimator
import bigdl.orca.automl.hp as hp
from bigdl.chronos.model.prophet import ProphetBuilder


# -

class AutoProphet:

    def __init__(self,
                 changepoint_prior_scale=hp.loguniform(0.001, 0.5),
                 seasonality_prior_scale=hp.loguniform(0.01, 10),
                 holidays_prior_scale=hp.loguniform(0.01, 10),
                 seasonality_mode=hp.choice(['additive', 'multiplicative']),
                 changepoint_range=hp.uniform(0.8, 0.95),
                 metric='mse',
                 logs_dir="/tmp/auto_prophet_logs",
                 cpus_per_trial=1,
                 name="auto_prophet",
                 remote_dir=None,
                 **prophet_config
                 ):
        """
        Create an automated Prophet Model.
        User need to specify either the exact value or the search space of the
        Prophet model hyperparameters. For details of the Prophet model hyperparameters, refer to
        https://facebook.github.io/prophet/docs/diagnostics.html#hyperparameter-tuning.

        :param changepoint_prior_scale: Int or hp sampling function from an integer space
            for hyperparameter changepoint_prior_scale for the Prophet model.
            For hp sampling, see bigdl.chronos.orca.automl.hp for more details.
            e.g. hp.loguniform(0.001, 0.5).
        :param seasonality_prior_scale: hyperparameter seasonality_prior_scale for the
            Prophet model.
            e.g. hp.loguniform(0.01, 10).
        :param holidays_prior_scale: hyperparameter holidays_prior_scale for the
            Prophet model.
            e.g. hp.loguniform(0.01, 10).
        :param seasonality_mode: hyperparameter seasonality_mode for the
            Prophet model.
            e.g. hp.choice(['additive', 'multiplicative']).
        :param changepoint_range: hyperparameter changepoint_range for the
            Prophet model.
            e.g. hp.uniform(0.8, 0.95).
        :param metric: String. The evaluation metric name to optimize. e.g. "mse"
        :param logs_dir: Local directory to save logs and results. It defaults to
            "/tmp/auto_prophet_logs"
        :param cpus_per_trial: Int. Number of cpus for each trial. It defaults to 1.
        :param name: name of the AutoProphet. It defaults to "auto_prophet"
        :param remote_dir: String. Remote directory to sync training results and checkpoints. It
            defaults to None and doesn't take effects while running in local. While running in
            cluster, it defaults to "hdfs:///tmp/{name}".

        :param prophet_config: Other Prophet hyperparameters.
        """
        self.search_space = {
            "changepoint_prior_scale": changepoint_prior_scale,
            "seasonality_prior_scale": seasonality_prior_scale,
            "holidays_prior_scale": holidays_prior_scale,
            "seasonality_mode": seasonality_mode,
            "changepoint_range": changepoint_range
        }
        self.search_space.update(prophet_config)  # update other configs
        self.metric = metric
        model_builder = ProphetBuilder()
        self.auto_est = AutoEstimator(model_builder=model_builder,
                                      logs_dir=logs_dir,
                                      resources_per_trial={"cpu": cpus_per_trial},
                                      remote_dir=remote_dir,
                                      name=name)

    def fit(self,
            data,
            cross_validation=True,
            expect_horizon=None,
            freq=None,
            metric_threshold=None,
            n_sampling=50,
            search_alg=None,
            search_alg_params=None,
            scheduler=None,
            scheduler_params=None,
            ):
        """
        Automatically fit the model and search for the best hyperparameters.

        :param data: training data, a pandas dataframe with Td rows,
               and 2 columns, with column 'ds' indicating date and column 'y' indicating value
               and Td is the time dimension
        :param cross_validation: bool, if the eval result comes from cross_validation.
               The value is set to True by default. Setting this option to False to
               speed up the process.
        :param expect_horizon: int, validation data will be automatically splited from training
               data, and expect_horizon is the horizon you may need to use once the mode is fitted.
               The value defaults to None, where 10% of training data will be taken
               as the validation data.
        :param freq: the freqency of the training dataframe. the frequency can be anything from the
               pandas list of frequency strings here:
               https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliasesDefaulted
               to None, where an unreliable frequency will be infer implicitly.
        :param metric_threshold: a trial will be terminated when metric threshold is met
        :param n_sampling: Number of times to sample from the search_space. Defaults to 50.
               If hp.grid_search is in search_space, the grid will be repeated n_sampling of times.
               If this is -1, (virtually) infinite samples are generated
               until a stopping condition is met.
        :param search_alg: str, all supported searcher provided by ray tune
               (i.e."variant_generator", "random", "ax", "dragonfly", "skopt",
               "hyperopt", "bayesopt", "bohb", "nevergrad", "optuna", "zoopt" and
               "sigopt")
        :param search_alg_params: extra parameters for searcher algorithm besides search_space,
               metric and searcher mode
        :param scheduler: str, all supported scheduler provided by ray tune
        :param scheduler_params: parameters for scheduler
        """
        if expect_horizon is None:
            expect_horizon = int(0.1*len(data))
        if freq is None:
            assert len(data) >= 2, "The training dataframe should contains more than 2 records."
            assert pd.api.types.is_datetime64_any_dtype(data["ds"].dtypes), \
                "The \"ds\" col should be in datetime 64 type, or you need to set `freq` in fit."
            self._freq = data["ds"].iloc[1] - data["ds"].iloc[0]
        else:
            self._freq = pd.Timedelta(freq)
        expect_horizon_str = str(self._freq * expect_horizon)
        self.search_space.update({"expect_horizon": expect_horizon_str,
                                  "cross_validation": cross_validation})
        train_data = data if cross_validation else data[:len(data)-expect_horizon]
        validation_data = None if cross_validation else data[len(data)-expect_horizon:]
        self.auto_est.fit(data=train_data,
                          validation_data=validation_data,
                          metric=self.metric,
                          metric_threshold=metric_threshold,
                          n_sampling=n_sampling,
                          search_space=self.search_space,
                          search_alg=search_alg,
                          search_alg_params=search_alg_params,
                          scheduler=scheduler,
                          scheduler_params=scheduler_params
                          )
        # use the best config to fit a new prophet model on whole data
        self.best_model = ProphetBuilder().build(self.auto_est.get_best_config())
        self.best_model.model.fit(data)

    def predict(self, horizon=1, freq="D", ds_data=None):
        """
        Predict using the best model after HPO.

        :param horizon: the number of steps forward to predict
        :param freq: the freqency of the predicted dataframe, defaulted to day("D"),
               the frequency can be anything from the pandas list of frequency strings here:
               https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
        :param ds_data: a dataframe that has 1 column 'ds' indicating date.
        """
        if self.best_model.model is None:
            raise RuntimeError(
                "You must call fit or restore first before calling predict!")
        return self.best_model.predict(horizon=horizon, freq=freq, ds_data=ds_data)

    def evaluate(self, data, metrics=['mse']):
        """
        Evaluate using the best model after HPO.

        :param data: evaluation data, a pandas dataframe with Td rows,
            and 2 columns, with column 'ds' indicating date and column 'y' indicating value
            and Td is the time dimension
        :param metrics: A list contains metrics for test/valid data.
        """
        if data is None:
            raise ValueError("Input invalid data of None")
        if self.best_model.model is None:
            raise RuntimeError(
                "You must call fit or restore first before calling evaluate!")
        return self.best_model.evaluate(target=data,
                                        metrics=metrics)

    def save(self, checkpoint_file):
        """
        Save the best model after HPO.

        :param checkpoint_file: The location you want to save the best model, should be a json file
        """
        if self.best_model.model is None:
            raise RuntimeError(
                "You must call fit or restore first before calling save!")
        self.best_model.save(checkpoint_file)

    def restore(self, checkpoint_file):
        """
        Restore the best model after HPO.

        :param checkpoint_file: The checkpoint file location you want to load the best model.
        """
        self.best_model.restore(checkpoint_file)

    def get_best_model(self):
        """
        Get the best Prophet model.
        """
        return self.best_model.model
