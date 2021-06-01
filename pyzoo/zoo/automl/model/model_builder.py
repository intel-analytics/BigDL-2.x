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
import logging

logger = logging.getLogger(__name__)


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


class XGBoostModelBuilder(ModelBuilder):

    def __init__(self, model_type="regressor", cpus_per_trial=1, **xgb_configs):
        self.model_type = model_type
        self.model_config = xgb_configs.copy()

        if 'n_jobs' in xgb_configs and xgb_configs['n_jobs'] != cpus_per_trial:
            logger.warning(f"Found n_jobs={xgb_configs['n_jobs']} in xgb_configs. It will not take "
                           f"effect since we assign cpus_per_trials(={cpus_per_trial}) to xgboost "
                           f"n_jobs. Please raise an issue if you do need different values for "
                           f"xgboost n_jobs and cpus_per_trials.")
        self.model_config['n_jobs'] = cpus_per_trial

    def build(self, config):
        from zoo.orca.automl.xgboost.XGBoost import XGBoost
        model = XGBoost(model_type=self.model_type, config=self.model_config)
        model._build(**config)
        return model

    def build_from_ckpt(self, checkpoint_filename):
        from zoo.orca.automl.xgboost.XGBoost import XGBoost
        model = XGBoost(model_type=self.model_type, config=self.model_config)
        model.restore(checkpoint_filename)
        return model
