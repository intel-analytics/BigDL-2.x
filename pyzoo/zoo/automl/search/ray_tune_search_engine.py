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

import ray
from ray import tune
from copy import deepcopy

from zoo.automl.search.base import *
from zoo.automl.common.util import *
from zoo.automl.common.metrics import Evaluator
from ray.tune import Trainable
import ray.tune.track
from zoo.automl.logger import TensorboardXLogger
from zoo.automl.model import ModelBuilder
from zoo.orca.automl import hp
from zoo.zouwu.feature.identity_transformer import IdentityTransformer
from zoo.zouwu.preprocessing.impute import LastFillImpute, FillZeroImpute
import pandas as pd

SEARCH_ALG_ALLOWED = ("variant_generator", "skopt", "bayesopt", "sigopt")


class RayTuneSearchEngine(SearchEngine):
    """
    Tune driver
    """

    def __init__(self,
                 logs_dir="",
                 resources_per_trial=None,
                 name="",
                 remote_dir=None,
                 ):
        """
        Constructor
        :param resources_per_trial: resources for each trial
        """
        self.pipeline = None
        self.train_func = None
        self.trainable_class = None
        self.resources_per_trail = resources_per_trial
        self.trials = None
        self.remote_dir = remote_dir
        self.name = name
        self.logs_dir = os.path.abspath(os.path.expanduser(logs_dir))

    def compile(self,
                data,
                model_create_func,
                recipe,
                search_space=None,
                search_alg=None,
                search_alg_params=None,
                scheduler=None,
                scheduler_params=None,
                feature_transformers=None,
                mc=False,
                metric="mse"):
        """
        Do necessary preparations for the engine
        :param input_df:
        :param search_space:
        :param num_samples:
        :param stop:
        :param search_algorithm:
        :param search_algorithm_params:
        :param fixed_params:
        :param feature_transformers:
        :param model:
        :param validation_df:
        :param metric:
        :return:
        """

        # data mode detection
        assert isinstance(data, dict), 'ERROR: Argument \'data\' should be a dictionary.'
        data_mode = None  # data_mode can only be 'dataframe' or 'ndarray'
        data_schema = set(data.keys())
        if set(["df"]).issubset(data_schema):
            data_mode = 'dataframe'
        if set(["x", "y"]).issubset(data_schema):
            data_mode = 'ndarray'
        assert data_mode in ['dataframe', 'ndarray'],\
            'ERROR: Argument \'data\' should fit either \
                dataframe schema (include \'df\' in keys) or\
                     ndarray (include \'x\' and \'y\' in keys) schema.'

        # data extract
        if data_mode == 'dataframe':
            input_df = data['df']
            feature_cols = data.get("feature_cols", None)
            target_col = data.get("target_col", None)
            validation_df = data.get("val_df", None)
        else:
            if data["x"].ndim == 1:
                data["x"] = data["x"].reshape(-1, 1)
            if data["y"].ndim == 1:
                data["y"] = data["y"].reshape(-1, 1)
            if "val_x" in data.keys() and data["val_x"].ndim == 1:
                data["val_x"] = data["val_x"].reshape(-1, 1)
            if "val_y" in data.keys() and data["val_y"].ndim == 1:
                data["val_y"] = data["val_y"].reshape(-1, 1)

            input_data = {"x": data["x"], "y": data["y"]}
            if 'val_x' in data.keys():
                validation_data = {"x": data["val_x"], "y": data["val_y"]}
            else:
                validation_data = None

        # prepare parameters for search engine
        runtime_params = recipe.runtime_params()
        self.num_samples = runtime_params['num_samples']
        stop = dict(runtime_params)
        del stop['num_samples']
        self.stop_criteria = stop
        if search_space is None:
            search_space = recipe.search_space(all_available_features=None)
        self._search_alg = RayTuneSearchEngine._set_search_alg(search_alg, search_alg_params,
                                                               recipe, search_space)
        self._scheduler = RayTuneSearchEngine._set_scheduler(scheduler, scheduler_params)
        self.search_space = self._prepare_tune_config(search_space)

        if feature_transformers is None and data_mode == 'dataframe':
            feature_transformers = IdentityTransformer(feature_cols, target_col)

        if data_mode == 'dataframe':
            self.train_func = self._prepare_train_func(input_data=input_df,
                                                       model_create_func=model_create_func,
                                                       feature_transformers=feature_transformers,
                                                       validation_data=validation_df,
                                                       metric=metric,
                                                       mc=mc,
                                                       remote_dir=self.remote_dir,
                                                       numpy_format=False
                                                       )
        else:
            self.train_func = self._prepare_train_func(input_data=input_data,
                                                       model_create_func=model_create_func,
                                                       feature_transformers=None,
                                                       validation_data=validation_data,
                                                       metric=metric,
                                                       mc=mc,
                                                       remote_dir=self.remote_dir,
                                                       numpy_format=True
                                                       )
        # self.trainable_class = self._prepare_trainable_class(input_df,
        #                                                      feature_transformers,
        #                                                      # model,
        #                                                      future_seq_len,
        #                                                      validation_df,
        #                                                      metric_op,
        #                                                      self.remote_dir)

    @staticmethod
    def _set_search_alg(search_alg, search_alg_params, recipe, search_space):
        if search_alg:
            if not isinstance(search_alg, str):
                raise ValueError(f"search_alg should be of type str."
                                 f" Got {search_alg.__class__.__name__}")
            search_alg = search_alg.lower()
            if search_alg_params is None:
                search_alg_params = dict()
            if search_alg not in SEARCH_ALG_ALLOWED:
                raise ValueError(f"search_alg must be one of {SEARCH_ALG_ALLOWED}. "
                                 f"Got: {search_alg}")
            elif search_alg == "bayesopt":
                search_alg_params.update({"space": recipe.manual_search_space()})

            search_alg_params.update(dict(
                metric="reward_metric",
                mode="max",
            ))
            search_alg = tune.create_searcher(search_alg, **search_alg_params)
        return search_alg

    @staticmethod
    def _set_scheduler(scheduler, scheduler_params):
        if scheduler:
            if not isinstance(scheduler, str):
                raise ValueError(f"Scheduler should be of type str. "
                                 f"Got {scheduler.__class__.__name__}")
            if scheduler_params is None:
                scheduler_params = dict()
            scheduler_params.update(dict(
                time_attr="training_iteration",
                metric="reward_metric",
                mode="max",
            ))
            scheduler = tune.create_scheduler(scheduler, **scheduler_params)
        return scheduler

    def run(self):
        """
        Run trials
        :return: trials result
        """
        analysis = tune.run(
            self.train_func,
            local_dir=self.logs_dir,
            name=self.name,
            stop=self.stop_criteria,
            config=self.search_space,
            search_alg=self._search_alg,
            num_samples=self.num_samples,
            scheduler=self._scheduler,
            resources_per_trial=self.resources_per_trail,
            verbose=1,
            reuse_actors=True
        )
        self.trials = analysis.trials

        # Visualization code for ray (leaderboard)
        # catch the ImportError Since it has been processed in TensorboardXLogger
        tf_config, tf_metric = self._log_adapt(analysis)

        self.logger = TensorboardXLogger(os.path.join(self.logs_dir, self.name+"_leaderboard"))
        self.logger.run(tf_config, tf_metric)
        self.logger.close()

        return analysis

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
            self.train_func({'out_units': 1,
                             'selected_features': ["MONTH(datetime)", "WEEKDAY(datetime)"]},
                            mock_reporter)
            # self.train_func(self.search_space, mock_reporter)

        except TypeError as e:
            print("Forgot to modify function signature?")
            raise e
        except GoodError:
            print("Works!")
            return 1
        raise Exception("Didn't call reporter...")

    @staticmethod
    def _is_validation_df_valid(validation_df):
        df_not_empty = isinstance(validation_df, pd.DataFrame) and not validation_df.empty
        df_list_not_empty = isinstance(validation_df, list) and validation_df \
            and not all([d.empty for d in validation_df])
        return validation_df is not None and not (df_not_empty or df_list_not_empty)

    @staticmethod
    def _prepare_train_func(input_data,
                            model_create_func,
                            feature_transformers,
                            metric,
                            validation_data=None,
                            mc=False,
                            remote_dir=None,
                            numpy_format=False,
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
        numpy_format_id = ray.put(numpy_format)
        input_data_id = ray.put(input_data)
        ft_id = ray.put(feature_transformers)

        # model_id = ray.put(model)

        # validation data processing
        df_not_empty = isinstance(validation_data, dict) or\
            (isinstance(validation_data, pd.DataFrame) and not validation_data.empty)
        df_list_not_empty = isinstance(validation_data, dict) or\
            (isinstance(validation_data, list) and validation_data
                and not all([d.empty for d in validation_data]))
        if validation_data is not None and (df_not_empty or df_list_not_empty):
            validation_data_id = ray.put(validation_data)
            is_val_valid = True
        else:
            is_val_valid = False

        def train_func(config):
            numpy_format = ray.get(numpy_format_id)

            if isinstance(model_create_func, ModelBuilder):
                trial_model = model_create_func.build(config)
            else:
                trial_model = model_create_func()

            if not numpy_format:
                global_ft = ray.get(ft_id)
                trial_ft = deepcopy(global_ft)
                imputer = None
                if "imputation" in config:
                    if config["imputation"] == "LastFillImpute":
                        imputer = LastFillImpute()
                    elif config["imputation"] == "FillZeroImpute":
                        imputer = FillZeroImpute()

                # handling input
                global_input_df = ray.get(input_data_id)
                trial_input_df = deepcopy(global_input_df)
                if imputer:
                    trial_input_df = imputer.impute(trial_input_df)
                config = convert_bayes_configs(config).copy()
                # print("config is ", config)
                (x_train, y_train) = trial_ft.fit_transform(trial_input_df, **config)
                # trial_ft.fit(trial_input_df, **config)

                # handling validation data
                validation_data = None
                if is_val_valid:
                    global_validation_df = ray.get(validation_data_id)
                    trial_validation_df = deepcopy(global_validation_df)
                    validation_data = trial_ft.transform(trial_validation_df)
            else:
                train_data = ray.get(input_data_id)
                x_train, y_train = (train_data["x"], train_data["y"])
                validation_data = None
                if is_val_valid:
                    validation_data = ray.get(validation_data_id)
                    validation_data = (validation_data["x"], validation_data["y"])
                trial_ft = None

            # no need to call build since it is called the first time fit_eval is called.
            # callbacks = [TuneCallback(tune_reporter)]
            # fit model
            best_reward_m = None
            # print("config:", config)
            for i in range(1, 101):
                result = trial_model.fit_eval(x_train,
                                              y_train,
                                              validation_data=validation_data,
                                              mc=mc,
                                              metric=metric,
                                              # verbose=1,
                                              **config)
                reward_m = result if Evaluator.get_metric_mode(metric) == "max" else -result
                checkpoint_filename = "best.ckpt"
                if isinstance(model_create_func, ModelBuilder):
                    trial_model.save(checkpoint_filename)
                else:
                    if best_reward_m is None or reward_m > best_reward_m:
                        best_reward_m = reward_m
                        save_zip(checkpoint_filename, trial_ft, trial_model, config)
                        if remote_dir is not None:
                            upload_ppl_hdfs(remote_dir, checkpoint_filename)

                tune.track.log(training_iteration=i,
                               reward_metric=reward_m,
                               checkpoint=checkpoint_filename)

        return train_func

    @staticmethod
    def _prepare_trainable_class(input_df,
                                 feature_transformers,
                                 future_seq_len,
                                 metric,
                                 validation_df=None,
                                 mc=False,
                                 remote_dir=None
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
        # model_id = ray.put(model)

        df_not_empty = isinstance(validation_df, pd.DataFrame) and not validation_df.empty
        df_list_not_empty = isinstance(validation_df, list) and validation_df \
            and not all([d.empty for d in validation_df])
        if validation_df is not None and (df_not_empty or df_list_not_empty):
            validation_df_id = ray.put(validation_df)
            is_val_df_valid = True
        else:
            is_val_df_valid = False

        class TrainableClass(Trainable):

            def _setup(self, config):
                # print("config in set up is", config)
                global_ft = ray.get(ft_id)
                # global_model = ray.get(model_id)
                self.trial_ft = deepcopy(global_ft)
                self.trial_model = TimeSequenceModel(check_optional_config=False,
                                                     future_seq_len=future_seq_len)

                # handling input
                global_input_df = ray.get(input_df_id)
                trial_input_df = deepcopy(global_input_df)
                self.config = convert_bayes_configs(config).copy()
                (self.x_train, self.y_train) = self.trial_ft.fit_transform(trial_input_df,
                                                                           **self.config)
                # trial_ft.fit(trial_input_df, **config)

                # handling validation data
                self.validation_data = None
                if is_val_df_valid:
                    global_validation_df = ray.get(validation_df_id)
                    trial_validation_df = deepcopy(global_validation_df)
                    self.validation_data = self.trial_ft.transform(trial_validation_df)

                # no need to call build since it is called the first time fit_eval is called.
                # callbacks = [TuneCallback(tune_reporter)]
                # fit model
                self.best_reward_m = -999
                self.reward_m = -999
                self.ckpt_name = "pipeline.ckpt"
                self.metric_op = 1 if metric_mode is "max" else -1

            def _train(self):
                # print("self.config in train is ", self.config)
                result = self.trial_model.fit_eval(self.x_train, self.y_train,
                                                   validation_data=self.validation_data,
                                                   # verbose=1,
                                                   **self.config)
                self.reward_m = result if Evaluator.get_metric_mode(metric) == "max" else -result
                # if metric == "mean_squared_error":
                #     self.reward_m = (-1) * result
                #     # print("running iteration: ",i)
                # elif metric == "r_square":
                #     self.reward_m = result
                # else:
                #     raise ValueError("metric can only be \"mean_squared_error\" or \"r_square\"")
                return {"reward_metric": self.reward_m, "checkpoint": self.ckpt_name}

            def _save(self, checkpoint_dir):
                # print("checkpoint dir is ", checkpoint_dir)
                ckpt_name = self.ckpt_name
                # save in the working dir (without "checkpoint_{}".format(training_iteration))
                path = os.path.join(checkpoint_dir, "..", ckpt_name)
                # path = os.path.join(checkpoint_dir, ckpt_name)
                # print("checkpoint save path is ", checkpoint_dir)
                if self.reward_m > self.best_reward_m:
                    self.best_reward_m = self.reward_m
                    print("****this reward is", self.reward_m)
                    print("*********saving checkpoint")
                    save_zip(ckpt_name, self.trial_ft, self.trial_model, self.config)
                    if remote_dir is not None:
                        upload_ppl_hdfs(remote_dir, ckpt_name)
                return path

            def _restore(self, checkpoint_path):
                # print("checkpoint path in restore is ", checkpoint_path)
                if remote_dir is not None:
                    restore_hdfs(checkpoint_path, remote_dir, self.trial_ft, self.trial_model)
                else:
                    restore_zip(checkpoint_path, self.trial_ft, self.trial_model)

        return TrainableClass

    def _prepare_tune_config(self, space):
        tune_config = {}
        for k, v in space.items():
            if isinstance(v, RandomSample):
                tune_config[k] = hp.sample_from(v.func)
            elif isinstance(v, GridSearch):
                tune_config[k] = hp.grid_search(v.values)
            else:
                tune_config[k] = v
        return tune_config

    def _log_adapt(self, analysis):
        # config
        config = analysis.get_all_configs()
        # metric
        metric_raw = analysis.fetch_trial_dataframes()
        metric = {}
        for key, value in metric_raw.items():
            metric[key] = dict(zip(list(value.columns), list(map(list, value.values.T))))
            config[key]["address"] = key
        return config, metric
