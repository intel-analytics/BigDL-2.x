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
from zoo.orca.learn.pytorch.training_operator import TrainingOperator


class Estimator(ABC):
    @abstractmethod
    def fit(self, data, epochs, **kwargs):
        pass

    @abstractmethod
    def predict(self, data, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, data, **kwargs):
        pass

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def save(self, checkpoint):
        pass

    @abstractmethod
    def load(self, checkpoint):
        pass

    @abstractmethod
    def shutdown(self, **kwargs):
        pass

    @staticmethod
    def from_torch(*,
                   model,
                   optimizer,
                   loss=None,
                   scheduler_creator=None,
                   training_operator_cls=TrainingOperator,
                   initialization_hook=None,
                   config=None,
                   scheduler_step_freq="batch",
                   use_tqdm=False,
                   workers_per_node=1,
                   backend="horovod"):
        from zoo.orca.learn.pytorch.estimator import PyTorchRayEstimatorWrapper
        if backend in {"horovod", "torch_distributed"}:
            return PyTorchRayEstimatorWrapper(model_creator=model,
                                              optimizer_creator=optimizer,
                                              loss_creator=loss,
                                              scheduler_creator=scheduler_creator,
                                              training_operator_cls=training_operator_cls,
                                              initialization_hook=initialization_hook,
                                              config=config,
                                              scheduler_step_freq=scheduler_step_freq,
                                              use_tqdm=use_tqdm,
                                              workers_per_node=workers_per_node,
                                              backend=backend)
        else:
            raise ValueError("only horovod and pytorch backend are supported for now,"
                             f" got backend: {backend}")

    @staticmethod
    def from_tf2(*,
                 model_creator,
                 compile_args_creator=None,
                 config=None,
                 verbose=False,
                 workers_per_node=1,
                 backend="tf2"):
        """Sets up the TensorFlow trainer.

        Args:
            model_creator (dict -> Model): This function takes in the `config`
                dict and returns a compiled TF model.
            compile_args_creator
            config (dict): configuration passed to 'model_creator',
                'data_creator'. Also contains `fit_config`, which is passed
                into `model.fit(data, **fit_config)` and
                `evaluate_config` which is passed into `model.evaluate`.
            verbose (bool): Prints output of one model if true.
            workers_per_node (int): worker number on each node. default: 1.
            backend (string): You can choose "horovod" or "tf2" as backend. Default: tf2.
        """
        from zoo.orca.learn.tf2.tf_ray_estimator import Estimator
        return Estimator(model_creator=model_creator,
                         compile_args_creator=compile_args_creator,
                         config=config,
                         verbose=verbose,
                         backend=backend,
                         workers_per_node=workers_per_node)

    @staticmethod
    def from_keras(*,
                   model_creator,
                   compile_args_creator=None,
                   config=None,
                   verbose=False,
                   workers_per_node=1,
                   backend="tf2"):
        """Sets up the TensorFlow trainer.

        Args:
            model_creator (dict -> Model): This function takes in the `config`
                dict and returns a compiled TF model.
            compile_args_creator
            config (dict): configuration passed to 'model_creator',
                'data_creator'. Also contains `fit_config`, which is passed
                into `model.fit(data, **fit_config)` and
                `evaluate_config` which is passed into `model.evaluate`.
            verbose (bool): Prints output of one model if true.
            workers_per_node (int): worker number on each node. default: 1.
            backend (string): You can choose "horovod" or "tf2" as backend. Default: tf2.
        """
        from zoo.orca.learn.tf2.tf_ray_estimator import Estimator
        return Estimator.from_keras(model_creator=model_creator,
                                    config=config,
                                    verbose=verbose,
                                    workers_per_node=workers_per_node,
                                    compile_args_creator=compile_args_creator,
                                    backend=backend)
