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
from abc import abstractmethod


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
        from zoo.orca.automl.pytorch_utils import validate_pytorch_loss, validate_pytorch_optim
        self.model_creator = model_creator
        optimizer = validate_pytorch_optim(optimizer_creator)
        self.optimizer_creator = optimizer
        loss = validate_pytorch_loss(loss_creator)
        self.loss_creator = loss

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


