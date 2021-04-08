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

class AutoEstimator:
    def __init__(self, estimator, searcher):
        self.estimator = estimator
        self.searcher = searcher

    @staticmethod
    def from_torch(*,
                   model_creator,
                   optimizer,
                   loss,
                   logs_dir,
                   resources_per_trial,
                   name,
                   ):
        """
        Create an AutoEstimator for torch.

        :param model_creator: PyTorch model creator function.
        :param optimizer: PyTorch optimizer creator function or pytorch optimizer name (string).
        :param loss: PyTorch loss instance or PyTorch loss creator function
            or pytorch loss name (string).
        :param logs_dir: Local directory to save logs and results.
        :param resources_per_trial: Dict. resources for each trial. e.g. {"cpu": 2}.
        :param name: Name of the auto estimator.
        :return: an AutoEstimator object.
        """
        from zoo.orca.automl.pytorch_model_utils import validate_pytorch_loss, \
            validate_pytorch_optim
        from zoo.automl.model import ModelBuilder
        from zoo.automl.search import SearchEngineFactory
        loss = validate_pytorch_loss(loss)
        optimizer = validate_pytorch_optim(optimizer)
        estimator = ModelBuilder.from_pytorch(model_creator=model_creator,
                                              optimizer_creator=optimizer,
                                              loss_creator=loss)
        searcher = SearchEngineFactory.create_engine(backend="ray",
                                                     logs_dir=logs_dir,
                                                     resources_per_trial=resources_per_trial,
                                                     name=name)
        return AutoEstimator(estimator=estimator, searcher=searcher)

    @staticmethod
    def from_keras(*,
                   model_creator,
                   logs_dir,
                   resources_per_trial,
                   name,
                   ):
        """
        Create an AutoEstimator for tensorflow keras.

        :param model_creator: Tensorflow keras model creator function.
        :param logs_dir: Local directory to save logs and results.
        :param resources_per_trial: Dict. resources for each trial. e.g. {"cpu": 2}.
        :param name: Name of the auto estimator.
        :return: an AutoEstimator object.
        """
        from zoo.automl.model import ModelBuilder
        from zoo.automl.search import SearchEngineFactory
        estimator = ModelBuilder.from_tfkeras(model_creator=model_creator)
        searcher = SearchEngineFactory.create_engine(backend="ray",
                                                     logs_dir=logs_dir,
                                                     resources_per_trial=resources_per_trial,
                                                     name=name)
        return AutoEstimator(estimator=estimator, searcher=searcher)

    def fit(self,
            data,
            recipe=None,
            metric=None,
            search_alg=None,
            search_alg_params=None,
            scheduler=None,
            scheduler_params=None,
            ):

        self.searcher.compile(data=data,
                              model_create_func=self.estimator,
                              recipe=recipe,
                              metric=metric,
                              search_alg=search_alg,
                              search_alg_params=search_alg_params,
                              scheduler=scheduler,
                              scheduler_params=scheduler_params)
        analysis = self.searcher.run()
        return analysis



