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

import os
import numpy as np
import ray
import tempfile
import zipfile
import shutil
from ray import tune
from copy import copy, deepcopy

import tensorflow as tf

tf.keras.models.save_model

from zoo.automl.search.abstract import *


class RayTuneSearchEngine(SearchEngine):
    """
    Tune driver
    """

    def __init__(self, logs_dir="", ray_num_cpus=6, resources_per_trial=None):
        """
        Constructor
        :param ray_num_cpus: the total number of cpus for ray
        :param resources_per_trial: resources for each trial
        """
        self.pipeline = None
        self.train_func = None
        self.resources_per_trail = resources_per_trial
        self.trials = None
        ray.init(num_cpus=ray_num_cpus, include_webui=False, ignore_reinit_error=True)

    def compile(self,
                input_df,
                search_space,
                num_samples=1,
                stop=None,
                feature_transformers=None,
                model=None,
                validation_df=None,
                metric="mean_squared_error"):
        """
        Do necessary preparations for the engine
        :param input_df:
        :param search_space:
        :param num_samples:
        :param stop:
        :param feature_transformers:
        :param model:
        :param validation_df:
        :param metric:
        :return:
        """
        self.search_space = self._prepare_tune_config(search_space)
        self.stop_criteria = stop
        self.num_samples = num_samples
        self.train_func = self._prepare_train_func(input_df,
                                                   feature_transformers,
                                                   model,
                                                   validation_df,
                                                   metric)

    def run(self):
        """
        Run trials
        :return: trials result
        """
        trials = tune.run(
            self.train_func,
            stop=self.stop_criteria,
            config=self.search_space,
            num_samples=self.num_samples,
            resources_per_trial=self.resources_per_trail,
            verbose=1,
            reuse_actors=True
        )
        self.trials = trials
        return self

    def get_best_trials(self, k=1):
        sorted_trials = RayTuneSearchEngine._get_sorted_trials(self.trials, metric="reward_metric")
        best_trials = sorted_trials[:k]
        return [self._make_trial_output(t) for t in best_trials]

    def _make_trial_output(self, trial):
        return TrialOutput(config=trial.config,
                           model_path=os.path.join(trial.logdir, trial.last_result["checkpoint"]))

    @staticmethod
    def _get_best_trial(trial_list, metric):
        """Retrieve the best trial."""
        return max(trial_list, key=lambda trial: trial.last_result.get(metric, 0))

    @staticmethod
    def _get_sorted_trials(trial_list, metric):
        return sorted(
            trial_list,
            key=lambda trial: trial.last_result.get(metric, 0),
            reverse=True)

    @staticmethod
    def _get_best_result(trial_list, metric):
        """Retrieve the last result from the best trial."""
        return {metric: RayTuneSearchEngine._get_best_trial(trial_list, metric).last_result[metric]}

    def test_run(self):
        def mock_reporter(**kwargs):
            assert "reward_metric" in kwargs, "Did not report proper metric"
            assert "checkpoint" in kwargs, "Accidentally removed `checkpoint`?"
            raise GoodError("This works.")

        try:
            self.train_func({'out_units': 1, 'selected_features': ["MONTH(datetime)", "WEEKDAY(datetime)"]}, mock_reporter)
            # self.train_func(self.search_space, mock_reporter)

        except TypeError as e:
            print("Forgot to modify function signature?")
            raise e
        except GoodError:
            print("Works!")
            return 1
        raise Exception("Didn't call reporter...")

    @staticmethod
    def _prepare_train_func(input_df,
                            feature_transformers,
                            model,
                            validation_df=None,
                            metric="mean_squared_error"
                            ):
        """
        Prepare the train function for ray tune
        :param input_df: input dataframe
        :param feature_transformers: feature transformers
        :param model: model or model selector
        :param validation_df: validation dataframe
        :param metric: the rewarding metric
        :return: the train function
        """
        input_df_id = ray.put(input_df)
        ft_id = ray.put(feature_transformers)
        model_id = ray.put(model)

        if validation_df is not None and not validation_df.empty:
            validation_df_id = ray.put(validation_df)

        def train_func(config, tune_reporter):
            # make a copy from global variables for trial to make changes
            global_ft = ray.get(ft_id)
            global_model = ray.get(model_id)
            trial_ft = deepcopy(global_ft)
            trial_model = deepcopy(global_model)

            # handling input
            global_input_df = ray.get(input_df_id)
            trial_input_df = deepcopy(global_input_df)
            (x_train, y_train) = trial_ft.fit_transform(trial_input_df, **config)
            # trial_ft.fit(trial_input_df, **config)

            # handling validation data
            validation_data = None
            if validation_df is not None and not validation_df.empty:
                global_validation_df = ray.get(validation_df_id)
                trial_validation_df = deepcopy(global_validation_df)
                validation_data = trial_ft.transform(trial_validation_df)

            # no need to call build since it is called the first time fit_eval is called.
            # callbacks = [TuneCallback(tune_reporter)]
            # fit model
            best_reward_m = -999
            reward_m = -999
            for i in range(1, 101):
                result = trial_model.fit_eval(x_train, y_train, validation_data=validation_data, **config)
                if metric == "mean_squared_error":
                    reward_m = (-1) * result
                    # print("running iteration: ",i)
                elif metric == "r_square":
                    reward_m = result
                else:
                    raise ValueError("metric can only be \"mean_squared_error\" or \"r_square\"")
                if reward_m > best_reward_m:
                    best_reward_m = reward_m

                    dirname = tempfile.mkdtemp(prefix="automl_")
                    try:
                        model_path = os.path.join(dirname, "weights_tune.h5")
                        config_path = os.path.join(dirname, "local_config.json")

                        trial_model.save(model_path=model_path, config_path=config_path)
                        trial_ft.save(file_path=config_path)

                        with zipfile.ZipFile("all.zip", 'w') as f:
                            for dirpath, dirnames, filenames in os.walk(dirname):
                                for filename in filenames:
                                    f.write(os.path.join(dirpath, filename), filename)
                    finally:
                        shutil.rmtree(dirname)

                tune_reporter(
                    training_iteration=i,
                    reward_metric=reward_m,
                    checkpoint="all.zip"
                )

        return train_func

    def _prepare_tune_config(self, space):
        tune_config = {}
        for k, v in space.items():
            if isinstance(v, RandomSample):
                tune_config[k] = tune.sample_from(v.func)
            elif isinstance(v, GridSearch):
                tune_config[k] = tune.grid_search(v.values)
            else:
                tune_config[k] = v
        return tune_config
