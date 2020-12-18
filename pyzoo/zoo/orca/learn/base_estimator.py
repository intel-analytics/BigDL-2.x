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


class BaseEstimator(ABC):
    @abstractmethod
    def fit(self, data, epochs, batch_size, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, data, batch_size, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, data, batch_size, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_model(self):
        raise NotImplementedError

    @abstractmethod
    def save(self, model_path):
        raise NotImplementedError

    @abstractmethod
    def load(self, checkpoint, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def clear_gradient_clipping(self):
        raise NotImplementedError

    @abstractmethod
    def set_constant_gradient_clipping(self, min, max):
        raise NotImplementedError

    @abstractmethod
    def set_l2_norm_gradient_clipping(self, clip_norm):
        raise NotImplementedError

    @abstractmethod
    def get_train_summary(self, tag=None):
        raise NotImplementedError

    @abstractmethod
    def get_validation_summary(self, tag=None):
        raise NotImplementedError

    @abstractmethod
    def set_tensorboard(self, log_dir, app_name):
        raise NotImplementedError
