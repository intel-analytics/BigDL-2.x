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
from abc import abstractmethod
from zoo.orca.learn.base_estimator import BaseEstimator


class Estimator(BaseEstimator):
    @abstractmethod
    def fit(self, data, epochs, batch_size, **kwargs):
        pass

    @abstractmethod
    def predict(self, data, batch_size, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, data, batch_size, num_steps=None, **kwargs):
        pass

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def save(self, checkpoint):
        pass

    @abstractmethod
    def load(self, checkpoint, **kwargs):
        pass

    @abstractmethod
    def shutdown(self, **kwargs):
        pass

    @abstractmethod
    def clear_gradient_clipping(self):
        pass

    @abstractmethod
    def set_constant_gradient_clipping(self, min, max):
        pass

    @abstractmethod
    def set_l2_norm_gradient_clipping(self, clip_norm):
        pass

    @abstractmethod
    def get_train_summary(self, tag=None):
        pass

    @abstractmethod
    def get_validation_summary(self, tag=None):
        pass

    @abstractmethod
    def set_tensorboard(self, log_dir, app_name):
        pass
