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


from abc import ABC, abstractmethod
from ray.tune import Stopper


class GoodError(Exception):
    pass


class SearchEngine(ABC):
    """
    Abstract Base Search Engine class. For hyper paramter tuning.
    """

    @abstractmethod
    def run(self):
        """
        Run the trials with searched parameters
        :return:
        """
        pass

    @abstractmethod
    def get_best_trials(self, k):
        """
        Get the best trials from .
        :param k: trials to be selected
        :return: the config of best k trials
        """
        pass


class TrialOutput(object):
    def __init__(self, config, model_path):
        self.config = config
        self.model_path = model_path


# stopper
class TrialStopper(Stopper):
    def __init__(self, stop, metric, mode):
        self._mode = mode
        self._metric = metric
        self._stop = stop

    def __call__(self, trial_id, result):
        if self._metric in self._stop.keys():
            if self._mode == "max" and result[self._metric] >= self._stop[self._metric]:
                return True
            if self._mode == "min" and result[self._metric] <= self._stop[self._metric]:
                return True
        if "training_iteration" in self._stop.keys():
            if result["training_iteration"] >= self._stop["training_iteration"]:
                return True
        return False

    def stop_all(self):
        return False
