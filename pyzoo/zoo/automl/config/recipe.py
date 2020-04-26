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

    def fixed_params(self):
        return None

    def search_algorithm_params(self):
        return None

    def search_algorithm(self):
        return None

    def scheduler_params(self):
        pass


class SmokeRecipe(Recipe):
    """
    A very simple Recipe for smoke test that runs one epoch and one iteration
    with only 1 random sample.
    """

    def __init__(self):
        super(self.__class__, self).__init__()

    def search_space(self, all_available_features):
        return {
            "selected_features": all_available_features,
            "model": "LSTM",
            "lstm_1_units": RandomSample(lambda spec: np.random.choice([32, 64], size=1)[0]),
            "dropout_1": RandomSample(lambda spec: np.random.uniform(0.2, 0.5)),
            "lstm_2_units": RandomSample(lambda spec: np.random.choice([32, 64], size=1)[0]),
            "dropout_2": RandomSample(lambda spec: np.random.uniform(0.2, 0.5)),
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
            "selected_features": all_available_features,
            "model": "MTNet",
            "lr": 0.001,
            "batch_size": 16,
            "epochs": 1,
            "dropout": 0.2,
            "time_step": RandomSample(lambda spec: np.random.choice([3, 4], size=1)[0]),
            "filter_size": 2,
            "long_num": RandomSample(lambda spec: np.random.choice([3, 4], size=1)[0]),
            "ar_size": RandomSample(lambda spec: np.random.choice([2, 3], size=1)[0]),
            "past_seq_len": RandomSample(lambda spec: (spec.config.long_num + 1)
                                         * spec.config.time_step),
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
            past_seq_config = RandomSample(
                lambda spec: np.random.randint(
                    look_back[0], look_back[1] + 1, size=1)[0])
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
            "selected_features": RandomSample(
                lambda spec: np.random.choice(
                    all_available_features,
                    size=np.random.randint(
                        low=3, high=len(all_available_features), size=1),
                    replace=False)),

            # -------- model selection TODO add MTNet
            "model": RandomSample(lambda spec: np.random.choice(["LSTM", "Seq2seq"], size=1)[0]),

            # --------- Vanilla LSTM model parameters
            "lstm_1_units": GridSearch([16, 32]),
            "dropout_1": 0.2,
            "lstm_2_units": GridSearch([16, 32]),
            "dropout_2": RandomSample(lambda spec: np.random.uniform(0.2, 0.5)),

            # ----------- Seq2Seq model parameters
            "latent_dim": GridSearch([32, 64]),
            "dropout": RandomSample(lambda spec: np.random.uniform(0.2, 0.5)),

            # ----------- optimization parameters
            "lr": RandomSample(lambda spec: np.random.uniform(0.001, 0.01)),
            "batch_size": RandomSample(lambda spec:
                                       np.random.choice([32, 64], size=1, replace=False)[0]),
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
        self.lstm_1_units_config = RandomSample(
            lambda spec: np.random.choice(
                lstm_1_units, size=1)[0])
        self.lstm_2_units_config = GridSearch(lstm_2_units)
        self.dropout_2_config = RandomSample(
            lambda spec: np.random.uniform(0.2, 0.5))

        # -- optimization params
        self.lr = RandomSample(lambda spec: np.random.uniform(0.001, 0.01))
        self.batch_size = GridSearch(batch_size)
        self.epochs = epochs

    def search_space(self, all_available_features):
        return {
            # -------- feature related parameters
            "selected_features": RandomSample(
                lambda spec: np.random.choice(
                    all_available_features,
                    size=np.random.randint(
                        low=3, high=len(all_available_features), size=1),
                    replace=False)),

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
                 filter_size=[2, 4],
                 long_num=[3, 4],
                 ar_size=[2, 3],
                 batch_size=[32, 64]):
        """
        Constructor.
        :param num_rand_samples: number of hyper-param configurations sampled randomly
        :param training_iteration: no. of iterations for training (n epochs) in trials
        :param epochs: no. of epochs to train in each iteration
        :param time_step: random search candidates for model param "time_step"
        :param filter_size: random search candidates for model param "filter_size"
        :param long_num: random search candidates for model param "long_num"
        :param ar_size: random search candidates for model param "ar_size"
        :param batch_size: grid search candidates for batch size
        """
        super(self.__class__, self).__init__()
        # -- run time params
        self.num_samples = num_rand_samples
        self.training_iteration = training_iteration

        # -- optimization params
        self.lr = RandomSample(lambda spec: np.random.uniform(0.001, 0.01))
        self.batch_size = self.batch_size = GridSearch(batch_size)
        self.epochs = epochs

        # ---- model params
        self.dropout = RandomSample(
            lambda spec: np.random.uniform(0.2, 0.5))
        self.time_step = RandomSample(
            lambda spec: np.random.choice(time_step, size=1)[0])
        self.filter_size = RandomSample(
            lambda spec: np.random.choice(filter_size, size=1)[0])
        self.long_num = RandomSample(
            lambda spec: np.random.choice(long_num, size=1)[0])
        self.ar_size = RandomSample(
            lambda spec: np.random.choice(ar_size, size=1)[0])
        self.past_seq_len = RandomSample(
            lambda spec: (
                spec.config.long_num + 1) * spec.config.time_step)

    def search_space(self, all_available_features):
        return {
            "selected_features": RandomSample(
                lambda spec: np.random.choice(
                    all_available_features,
                    size=np.random.randint(
                        low=3, high=len(all_available_features), size=1),
                    replace=False)),

            "model": "MTNet",
            "lr": self.lr,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "dropout": self.dropout,
            "time_step": self.time_step,
            "filter_size": self.filter_size,
            "long_num": self.long_num,
            "ar_size": self.ar_size,
            "past_seq_len": self.past_seq_len,
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
        return {
            # -------- feature related parameters
            "selected_features": RandomSample(
                lambda spec: np.random.choice(
                    all_available_features,
                    size=np.random.randint(low=3, high=len(all_available_features), size=1))
            ),

            "model": RandomSample(lambda spec: np.random.choice(["LSTM", "Seq2seq"], size=1)[0]),
            # --------- Vanilla LSTM model parameters
            "lstm_1_units": RandomSample(lambda spec:
                                         np.random.choice([8, 16, 32, 64, 128], size=1)[0]),
            "dropout_1": RandomSample(lambda spec: np.random.uniform(0.2, 0.5)),
            "lstm_2_units": RandomSample(lambda spec:
                                         np.random.choice([8, 16, 32, 64, 128], size=1)[0]),
            "dropout_2": RandomSample(lambda spec: np.random.uniform(0.2, 0.5)),

            # ----------- Seq2Seq model parameters
            "latent_dim": RandomSample(lambda spec:
                                       np.random.choice([32, 64, 128, 256], size=1)[0]),
            "dropout": RandomSample(lambda spec: np.random.uniform(0.2, 0.5)),

            # ----------- optimization parameters
            "lr": RandomSample(lambda spec: np.random.uniform(0.001, 0.01)),
            "batch_size": RandomSample(lambda spec:
                                       np.random.choice([32, 64, 1024], size=1, replace=False)[0]),
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

    def search_space(self, all_available_features):
        feature_space = {"bayes_feature_{}".format(feature): (0.3, 1)
                         for feature in all_available_features}
        other_space = {
            # --------- model parameters
            "lstm_1_units_float": (8, 128),
            "dropout_1": (0.2, 0.5),
            "lstm_2_units_float": (8, 128),
            "dropout_2": (0.2, 0.5),

            # ----------- optimization parameters
            "lr": (0.001, 0.01),
            "batch_size_log": (5, 10),
        }
        total_space = other_space.copy()
        total_space.update(feature_space)
        total_space.update(self.bayes_past_seq_config)
        return total_space

    def fixed_params(self):
        total_fixed_params = {
            "epochs": self.epochs,
            # "batch_size": 1024,
        }
        total_fixed_params.update(self.fixed_past_seq_config)
        return total_fixed_params

    def search_algorithm_params(self):
        return {
            "utility_kwargs": {
                "kind": "ucb",
                "kappa": 2.5,
                "xi": 0.0
            }
        }

    def search_algorithm(self):
        return 'BayesOpt'
