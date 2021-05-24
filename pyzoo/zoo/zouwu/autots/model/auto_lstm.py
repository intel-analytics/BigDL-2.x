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
from zoo.automl.model import PytorchModelBuilder
from zoo.orca.automl.auto_estimator import AutoEstimator
from zoo.zouwu.model.VanillaLSTM_pytorch import model_creator


class AutoLSTM:
    def __init__(self,
                 input_dim,
                 output_dim,
                 optimizer,
                 loss,
                 metric,
                 hidden_dim=32,
                 layer_num=1,
                 lr=0.001,
                 dropout=0.2,
                 backend="torch",
                 logs_dir="/tmp/auto_lstm",
                 cpus_per_trial=1,
                 name="auto_lstm"):
        # todo: support backend = 'keras'
        if backend != "torch":
            raise ValueError(f"We only support backend as torch. Got {backend}")
        self.search_space = dict(
            hidden_dim=hidden_dim,
            layer_num=layer_num,
            lr=lr,
            dropout=dropout,
            input_feature_num=input_dim,
            output_feature_num=output_dim,
        )
        self.metric = metric
        model_builder = PytorchModelBuilder(model_creator=model_creator,
                                            optimizer_creator=optimizer,
                                            loss_creator=loss,
                                            )
        self.auto_est = AutoEstimator(model_builder=model_builder,
                                      logs_dir=logs_dir,
                                      resources_per_trial={"cpu": cpus_per_trial},
                                      name=name)

    def fit(self,
            data,
            epochs=1,
            batch_size=32,
            validation_data=None,
            metric_threshold=None,
            n_sampling=1,
            search_alg=None,
            search_alg_params=None,
            scheduler=None,
            scheduler_params=None,
            ):
        self.search_space["batch_size"] = batch_size
        self.auto_est.fit(
            data=data,
            epochs=epochs,
            validation_data=validation_data,
            metric=self.metric,
            metric_threshold=metric_threshold,
            n_sampling=n_sampling,
            search_space=self.search_space,
            search_alg=search_alg,
            search_alg_params=search_alg_params,
            scheduler=scheduler,
            scheduler_params=scheduler_params,
        )

    def get_best_model(self):
        return self.auto_est.get_best_model()
