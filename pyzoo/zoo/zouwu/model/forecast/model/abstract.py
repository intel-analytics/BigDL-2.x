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


class BaseModel(ABC):
    """
    base model for automl tuning
    """

    check_optional_config = False
    future_seq_len = None

    @abstractmethod
    def fit_eval(self, x, y, validation_data=None, mc=False, verbose=0, **config):
        """
        optimize and evaluate for one iteration for tuning
        :param config: tunable parameters for optimization
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, x, y, metric=None):
        """
        Evaluate the model
        :param x: input
        :param y: target
        :param metric:
        :return: a list of metric evaluation results
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, x):
        """
        Prediction.
        :param x: input
        :return: result
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, checkpoint_file):
        """
        save model to file.
        :param checkpoint_file: checkpoint file name to be saved to.
        :return:
        """
        pass

    @abstractmethod
    def restore(self, checkpoint_file):
        """
        restore model from model file.
        :param checkpoint_file: checkpoint file name to restore from.
        :return:
        """
        pass

    @abstractmethod
    def _get_required_parameters(self):
        """
        :return: required parameters to be set into config
        """
        return set()

    @abstractmethod
    def _get_optional_parameters(self):
        """
        :return: optional parameters to be set into config
        """
        return set()

    def _check_config(self, **config):
        """
        Do necessary checking for config
        :param config:
        :return:
        """
        config_parameters = set(config.keys())
        if not config_parameters.issuperset(self._get_required_parameters()):
            raise ValueError("Missing required parameters in configuration. " +
                             "Required parameters are: " + str(self._get_required_parameters()))
        if self.check_optional_config and \
                not config_parameters.issuperset(self._get_optional_parameters()):
            raise ValueError("Missing optional parameters in configuration. " +
                             "Optional parameters are: " + str(self._get_optional_parameters()))
        return True
