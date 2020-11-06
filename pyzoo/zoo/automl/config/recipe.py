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

from abc import ABCMeta, abstractmethod
from zoo.automl.search.abstract import *
import numpy as np
from ray import tune
import json


class Recipe(metaclass=ABCMeta):
    """
    Recipe
    """

    def __init__(self):
        # ----- runtime parameters
        self.training_iteration = 1
        self.num_samples = 1
        self.reward_metric = None

    @abstractmethod
    def search_space(self, all_available_features):
        pass

    def runtime_params(self):
        runtime_config = {
            "training_iteration": self.training_iteration,
            "num_samples": self.num_samples,
        }
        if self.reward_metric is not None:
            runtime_config["reward_metric"] = self.reward_metric
        return runtime_config

    def manual_search_space(self):
        return None


class SmokeRecipe(Recipe):
    """
    A very simple Recipe for smoke test that runs one epoch and one iteration
    with only 1 random sample.
    """

    def __init__(self):
        super(self.__class__, self).__init__()

    def search_space(self, all_available_features):
        return {
            "selected_features": json.dumps(all_available_features),
            "model": "LSTM",
            "lstm_1_units": tune.choice([32, 64]),
            "dropout_1": tune.uniform(0.2, 0.5),
            "lstm_2_units": tune.choice([32, 64]),
            "dropout_2": tune.uniform(0.2, 0.5),
            "lr": 0.001,
            "batch_size": 1024,
            "epochs": 1,
            "past_seq_len": 2,
        }


class MTNetSmokeRecipe(Recipe):
    """
    A very simple Recipe for smoke test that runs one epoch and one iteration
    with only 1 random sample.
    """

    def __init__(self):
        super(self.__class__, self).__init__()

    def search_space(self, all_available_features):
        return {
            "selected_features": json.dumps(all_available_features),
            "model": "MTNet",
            "lr": 0.001,
            "batch_size": 16,
            "epochs": 1,
            "cnn_dropout": 0.2,
            "rnn_dropout": 0.2,
            "time_step": tune.choice([3, 4]),
            "cnn_height": 2,
            "long_num": tune.choice([3, 4]),
            "ar_size": tune.choice([2, 3]),
            "past_seq_len": tune.sample_from(lambda spec:
                                             (spec.config.long_num + 1) * spec.config.time_step),
        }


class PastSeqParamHandler(object):
    """
    Utility to handle PastSeq Param
    """

    def __init__(self):
        pass

    @staticmethod
    def get_past_seq_config(look_back):
        """
        generate pass sequence config based on look_back
        :param look_back: look_back configuration
        :return: search configuration for past sequence
        """
        if isinstance(
            look_back,
            tuple) and len(look_back) == 2 and isinstance(
                look_back[0],
                int) and isinstance(
                look_back[1],
                int):
            if look_back[1] < 2:
                raise ValueError(
                    "The max look back value should be at least 2")
            if look_back[0] < 2:
                print(
                    "The input min look back value is smaller than 2. "
                    "We sample from range (2, {}) instead.".format(
                        look_back[1]))
            past_seq_config = tune.randint(look_back[0], look_back[1] + 1)
        elif isinstance(look_back, int):
            if look_back < 2:
                raise ValueError(
                    "look back value should not be smaller than 2. "
                    "Current value is ", look_back)
            past_seq_config = look_back
        else:
            raise ValueError(
                "look back is {}.\n "
                "look_back should be either a tuple with 2 int values:"
                " (min_len, max_len) or a single int".format(look_back))
        return past_seq_config


class GridRandomRecipe(Recipe):
    """
    A recipe involves both grid search and random search.
       tsp = TimeSequencePredictor(...,recipe = GridRandomRecipe(1))
    """

    def __init__(
            self,
            num_rand_samples=1,
            look_back=2,
            epochs=5,
            training_iteration=10):
        """
        Constructor.
        :param num_rand_samples: number of hyper-param configurations sampled randomly
        :param look_back: the length to look back, either a tuple with 2 int values,
          which is in format is (min len, max len), or a single int, which is
          a fixed length to look back.
        :param training_iteration: no. of iterations for training (n epochs) in trials
        :param epochs: no. of epochs to train in each iteration
        """
        super(self.__class__, self).__init__()
        self.num_samples = num_rand_samples
        self.training_iteration = training_iteration
        self.past_seq_config = PastSeqParamHandler.get_past_seq_config(
            look_back)
        self.epochs = epochs

    def search_space(self, all_available_features):
        return {
            # -------- feature related parameters
            "selected_features": tune.sample_from(lambda spec:
                                                  json.dumps(
                                                      list(np.random.choice(
                                                          all_available_features,
                                                          size=np.random.randint(
                                                              low=3,
                                                              high=len(all_available_features)),
                                                          replace=False)))),

            # -------- model selection TODO add MTNet
            "model": tune.choice(["LSTM", "Seq2seq"]),

            # --------- Vanilla LSTM model parameters
            "lstm_1_units": tune.grid_search([16, 32]),
            "dropout_1": 0.2,
            "lstm_2_units": tune.grid_search([16, 32]),
            "dropout_2": tune.uniform(0.2, 0.5),

            # ----------- Seq2Seq model parameters
            "latent_dim": tune.grid_search([32, 64]),
            "dropout": tune.uniform(0.2, 0.5),

            # ----------- optimization parameters
            "lr": tune.uniform(0.001, 0.01),
            "batch_size": tune.choice([32, 64], replace=False),
            "epochs": self.epochs,
            "past_seq_len": self.past_seq_config,
        }


class LSTMGridRandomRecipe(Recipe):
    """
    A recipe involves both grid search and random search, only for LSTM.
       tsp = TimeSequencePredictor(...,recipe = LSTMGridRandomRecipe(1))
    """

    def __init__(
            self,
            num_rand_samples=1,
            epochs=5,
            training_iteration=10,
            look_back=2,
            lstm_1_units=[16, 32, 64, 128],
            lstm_2_units=[16, 32, 64],
            batch_size=[32, 64]):
        """
        Constructor.
        :param lstm_1_units: random search candidates for num of lstm_1_units
        :param lstm_2_units: grid search candidates for num of lstm_1_units
        :param batch_size: grid search candidates for batch size
        :param num_rand_samples: number of hyper-param configurations sampled randomly
        :param look_back: the length to look back, either a tuple with 2 int values,
          which is in format is (min len, max len), or a single int, which is
          a fixed length to look back.
        :param training_iteration: no. of iterations for training (n epochs) in trials
        :param epochs: no. of epochs to train in each iteration
        """
        super(self.__class__, self).__init__()
        # -- runtime params
        self.num_samples = num_rand_samples
        self.training_iteration = training_iteration

        # -- model params
        self.past_seq_config = PastSeqParamHandler.get_past_seq_config(
            look_back)
        self.lstm_1_units_config = tune.choice(lstm_1_units)
        self.lstm_2_units_config = tune.grid_search(lstm_2_units)
        self.dropout_2_config = tune.uniform(0.2, 0.5)

        # -- optimization params
        self.lr = tune.uniform(0.001, 0.01)
        self.batch_size = tune.grid_search(batch_size)
        self.epochs = epochs

    def search_space(self, all_available_features):
        return {
            # -------- feature related parameters
            "selected_features": tune.sample_from(lambda spec:
                                                  json.dumps(
                                                      list(np.random.choice(
                                                          all_available_features,
                                                          size=np.random.randint(
                                                              low=3,
                                                              high=len(all_available_features) + 1),
                                                          replace=False)))),

            "model": "LSTM",

            # --------- Vanilla LSTM model parameters
            "lstm_1_units": self.lstm_1_units_config,
            "dropout_1": 0.2,
            "lstm_2_units": self.lstm_2_units_config,
            "dropout_2": self.dropout_2_config,

            # ----------- optimization parameters
            "lr": self.lr,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "past_seq_len": self.past_seq_config,
        }


class MTNetGridRandomRecipe(Recipe):
    """
    Grid+Random Recipe for MTNet
    """

    def __init__(self,
                 num_rand_samples=1,
                 epochs=5,
                 training_iteration=10,
                 time_step=[3, 4],
                 long_num=[3, 4],
                 cnn_height=[2, 3],
                 cnn_hid_size=[32, 50, 100],
                 ar_size=[2, 3],
                 batch_size=[32, 64]):
        """
        Constructor.
        :param num_rand_samples: number of hyper-param configurations sampled randomly
        :param training_iteration: no. of iterations for training (n epochs) in trials
        :param epochs: no. of epochs to train in each iteration
        :param time_step: random search candidates for model param "time_step"
        :param long_num: random search candidates for model param "long_num"
        :param ar_size: random search candidates for model param "ar_size"
        :param batch_size: grid search candidates for batch size
        :param cnn_height: random search candidates for model param "cnn_height"
        :param cnn_hid_size: random search candidates for model param "cnn_hid_size"
        """
        super(self.__class__, self).__init__()
        # -- run time params
        self.num_samples = num_rand_samples
        self.training_iteration = training_iteration

        # -- optimization params
        self.lr = tune.uniform(0.001, 0.01)
        self.batch_size = self.batch_size = tune.grid_search(batch_size)
        self.epochs = epochs

        # ---- model params
        self.cnn_dropout = tune.uniform(0.2, 0.5)
        self.rnn_dropout = tune.uniform(0.2, 0.5)
        self.time_step = tune.choice(time_step)
        self.long_num = tune.choice(long_num,)
        self.cnn_height = tune.choice(cnn_height)
        self.cnn_hid_size = tune.choice(cnn_hid_size)
        self.ar_size = tune.choice(ar_size)
        self.past_seq_len = tune.sample_from(
            lambda spec: (
                spec.config.long_num + 1) * spec.config.time_step)

    def search_space(self, all_available_features):
        return {
            "selected_features": tune.sample_from(lambda spec:
                                                  json.dumps(
                                                      list(np.random.choice(
                                                          all_available_features,
                                                          size=np.random.randint(
                                                              low=3,
                                                              high=len(all_available_features)),
                                                          replace=False)))),

            "model": "MTNet",
            "lr": self.lr,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "cnn_dropout": self.cnn_dropout,
            "rnn_dropout": self.rnn_dropout,
            "time_step": self.time_step,
            "long_num": self.long_num,
            "ar_size": self.ar_size,
            "past_seq_len": self.past_seq_len,
            "cnn_hid_size": self.cnn_hid_size,
            "cnn_height": self.cnn_height
        }


class RandomRecipe(Recipe):
    """
    Pure random sample Recipe. Often used as baseline.
       tsp = TimeSequencePredictor(...,recipe = RandomRecipe(5))
    """

    def __init__(
            self,
            num_rand_samples=1,
            look_back=2,
            epochs=5,
            reward_metric=-0.05,
            training_iteration=10):
        """
        :param num_rand_samples: number of hyper-param configurations sampled randomly
        :param look_back:the length to look back, either a tuple with 2 int values,
          which is in format is (min len, max len), or a single int, which is
          a fixed length to look back.
        :param reward_metric: the rewarding metric value, when reached, stop trial
        :param training_iteration: no. of iterations for training (n epochs) in trials
        :param epochs: no. of epochs to train in each iteration
        """
        super(self.__class__, self).__init__()
        self.num_samples = num_rand_samples
        self.reward_metric = reward_metric
        self.training_iteration = training_iteration
        self.epochs = epochs
        self.past_seq_config = PastSeqParamHandler.get_past_seq_config(
            look_back)

    def search_space(self, all_available_features):
        import random
        return {
            # -------- feature related parameters
            "selected_features": tune.sample_from(lambda spec:
                                                  json.dumps(
                                                      list(np.random.choice(
                                                          all_available_features,
                                                          size=np.random.randint(
                                                              low=3,
                                                              high=len(all_available_features)),
                                                          replace=False)))),

            "model": tune.choice(["LSTM", "Seq2seq"]),
            # --------- Vanilla LSTM model parameters
            "lstm_1_units": tune.choice([8, 16, 32, 64, 128]),
            "dropout_1": tune.uniform(0.2, 0.5),
            "lstm_2_units": tune.choice([8, 16, 32, 64, 128]),
            "dropout_2": tune.uniform(0.2, 0.5),

            # ----------- Seq2Seq model parameters
            "latent_dim": tune.choice([32, 64, 128, 256]),
            "dropout": tune.uniform(0.2, 0.5),

            # ----------- optimization parameters
            "lr": tune.uniform(0.001, 0.01),
            "batch_size": tune.choice([32, 64, 1024], replace=False),
            "epochs": self.epochs,
            "past_seq_len": self.past_seq_config,
        }


class BayesRecipe(Recipe):
    """
    A Bayes search Recipe. (Experimental)
       tsp = TimeSequencePredictor(...,recipe = BayesRecipe(5))
    """

    def __init__(
            self,
            num_samples=1,
            look_back=2,
            epochs=5,
            reward_metric=-0.05,
            training_iteration=5):
        """
        Constructor
        :param num_samples: number of hyper-param configurations sampled
        :param look_back: the length to look back, either a tuple with 2 int values,
          which is in format is (min len, max len), or a single int, which is
          a fixed length to look back.
        :param reward_metric: the rewarding metric value, when reached, stop trial
        :param training_iteration: no. of iterations for training (n epochs) in trials
        :param epochs: no. of epochs to train in each iteration
        """
        super(self.__class__, self).__init__()
        self.num_samples = num_samples
        self.reward_metric = reward_metric
        self.training_iteration = training_iteration
        self.epochs = epochs
        if isinstance(look_back, tuple) and len(look_back) == 2 and \
                isinstance(look_back[0], int) and isinstance(look_back[1], int):
            if look_back[1] < 2:
                raise ValueError("The max look back value should be at least 2")
            if look_back[0] < 2:
                print("The input min look back value is smaller than 2. "
                      "We sample from range (2, {}) instead.".format(look_back[1]))
            self.bayes_past_seq_config = {"past_seq_len_float": look_back}
            self.fixed_past_seq_config = {}
        elif isinstance(look_back, int):
            if look_back < 2:
                raise ValueError(
                    "look back value should not be smaller than 2. "
                    "Current value is ", look_back)
            self.bayes_past_seq_config = {}
            self.fixed_past_seq_config = {"past_seq_len": look_back}
        else:
            raise ValueError(
                "look back is {}.\n "
                "look_back should be either a tuple with 2 int values:"
                " (min_len, max_len) or a single int".format(look_back))

    def manual_search_space(self):
        model_space = {
            # --------- model parameters
            "lstm_1_units_float": (8, 128),
            "dropout_1": (0.2, 0.5),
            "lstm_2_units_float": (8, 128),
            "dropout_2": (0.2, 0.5),

            # ----------- optimization parameters
            "lr": (0.001, 0.01),
            "batch_size_log": (5, 10),
        }
        total_space = model_space.copy()
        total_space.update(self.bayes_past_seq_config)
        return total_space

    def search_space(self, all_available_features):
        total_fixed_params = {
            "epochs": self.epochs,
            "model": "LSTM",
            "selected_features": json.dumps(all_available_features),
            # "batch_size": 1024,
        }
        total_fixed_params.update(self.fixed_past_seq_config)
        return total_fixed_params


class XgbRegressorGridRandomRecipe(Recipe):
    def __init__(
            self,
            num_rand_samples=1,
            n_estimators=[8, 15],
            max_depth=[10, 15],
            n_jobs=-1,
            tree_method='hist',
            random_state=2,
            seed=0,
            lr=(1e-4, 1e-1),
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=[1, 2, 3],
            gamma=0,
            reg_alpha=0,
            reg_lambda=1):
        """
        """
        super(self.__class__, self).__init__()

        self.num_samples = num_rand_samples
        self.n_jobs = n_jobs
        self.tree_method = tree_method
        self.random_state = random_state
        self.seed = seed

        self.colsample_bytree = colsample_bytree
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda

        self.n_estimators = tune.grid_search(n_estimators)
        self.max_depth = tune.grid_search(max_depth)
        self.lr = tune.loguniform(lr[0], lr[-1])
        self.subsample = subsample
        self.min_child_weight = tune.choice(min_child_weight)

    def search_space(self, all_available_features):
        return {
            # -------- feature related parameters
            "model": "XGBRegressor",

            "imputation": tune.choice(["LastFillImpute", "FillZeroImpute"]),
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_child_weight": self.min_child_weight,
            "lr": self.lr
        }


class XgbRegressorSkOptRecipe(Recipe):
    def __init__(
            self,
            num_rand_samples=10,
            n_estimators_range=(50, 1000),
            max_depth_range=(2, 15),
    ):
        """
        """
        super(self.__class__, self).__init__()

        self.num_samples = num_rand_samples

        self.n_estimators_range = n_estimators_range
        self.max_depth_range = max_depth_range

    def search_space(self, all_available_features):
        space = {
            "n_estimators": tune.randint(self.n_estimators_range[0],
                                         self.n_estimators_range[1]),
            "max_depth": tune.randint(self.max_depth_range[0],
                                      self.max_depth_range[1]),
        }
        return space

    def opt_params(self):
        from skopt.space import Integer
        params = [
            Integer(self.n_estimators_range[0], self.n_estimators_range[1]),
            Integer(self.max_depth_range[0], self.max_depth_range[1]),
        ]
        return params
