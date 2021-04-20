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
from abc import ABC, abstractmethod


class ModelBuilder:

    @abstractmethod
    def build(self, config):
        pass


class KerasModelBuilder(ModelBuilder):

    def __init__(self, model_creator):
        self.model_creator = model_creator

    def build(self, config):
        from zoo.automl.model.base_keras_model import KerasBaseModel
        model = KerasBaseModel(self.model_creator)
        model.build(config)
        return model

    def build_from_ckpt(self, checkpoint_filename):
        from zoo.automl.model.base_keras_model import KerasBaseModel
        model = KerasBaseModel(self.model_creator)
        model.restore(checkpoint_filename)
        return model


class PytorchModelBuilder(ModelBuilder):

    def __init__(self, model_creator,
                 optimizer_creator,
                 loss_creator):
        self.model_creator = model_creator
        self.optimizer_creator = optimizer_creator
        self.loss_creator = loss_creator

    def build(self, config):
        from zoo.automl.model.base_pytorch_model import PytorchBaseModel
        model = PytorchBaseModel(self.model_creator,
                                 self.optimizer_creator,
                                 self.loss_creator)
        model.build(config)
        return model

    def build_from_ckpt(self, checkpoint_filename):
        '''Restore from a saved model'''
        from zoo.automl.model.base_pytorch_model import PytorchBaseModel
        model = PytorchBaseModel(self.model_creator,
                                 self.optimizer_creator,
                                 self.loss_creator)
        model.restore(checkpoint_filename)
        return model


class XGBoostModelBuilder(ModelBuilder):

    def __init__(self, model_type, n_cpus):
        self.model_type = model_type
        self.n_cpus = n_cpus

    def build(self, config):
        from zoo.orca.automl.xgboost.XGBoost import XGBoost
        model = XGBoost(model_type=self.model_type, config=config)

        if self.n_cpus is not None:
            model.set_params(n_jobs=self.n_cpus)
        return model
