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
import os
from unittest import TestCase

import numpy as np
import pytest

import torch
import torch.nn as nn
from zoo.orca.automl.auto_estimator import AutoEstimator
from zoo.automl.recipe.base import Recipe
from zoo.orca.automl.pytorch_utils import LR_NAME

os.environ["KMP_SETTINGS"] = "0"


class Net(nn.Module):
    def __init__(self, dropout, fc1_size, fc2_size):
        super().__init__()
        self.fc1 = nn.Linear(50, fc1_size)
        self.relu1 = nn.ReLU()
        self.dout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.prelu = nn.PReLU(1)
        self.out = nn.Linear(fc2_size, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, input_):
        a1 = self.fc1(input_)
        h1 = self.relu1(a1)
        dout = self.dout(h1)
        a2 = self.fc2(dout)
        h2 = self.prelu(a2)
        a3 = self.out(h2)
        y = self.out_act(a3)
        return y


def model_creator(config):
    return Net(dropout=config["dropout"],
               fc1_size=config["fc1_size"],
               fc2_size=config["fc2_size"])


def get_optimizer(model, config):
    return torch.optim.SGD(model.parameters(), lr=config["lr"])


def get_train_val_data():
    def get_x_y(size):
        input_size = 50
        x1 = np.random.randn(size // 2, input_size)
        x2 = np.random.randn(size // 2, input_size) + 1.5
        x = np.concatenate([x1, x2], axis=0)
        y1 = np.zeros((size // 2, 1))
        y2 = np.ones((size // 2, 1))
        y = np.concatenate([y1, y2], axis=0)
        return x, y
    train_x, train_y = get_x_y(size=1000)
    val_x, val_y = get_x_y(size=400)
    data = {'x': train_x, 'y': train_y, 'val_x': val_x, 'val_y': val_y}
    return data


class LinearRecipe(Recipe):
    def search_space(self, all_available_features):
        from zoo.orca.automl import hp
        return {
            "dropout": hp.uniform(0.2, 0.3),
            "fc1_size": hp.choice([50, 64]),
            "fc2_size": hp.choice([100, 128]),
            LR_NAME: hp.choice([0.001, 0.003, 0.01]),
            "batch_size": hp.choice([32, 64])
        }

    def runtime_params(self):
        return {
            "training_iteration": 1,
            "num_samples": 4
        }


class TestPyTorchAutoEstimator(TestCase):
    def setUp(self) -> None:
        from zoo.orca import init_orca_context
        init_orca_context(cores=8, init_ray_on_spark=True)

    def tearDown(self) -> None:
        from zoo.orca import stop_orca_context
        stop_orca_context()

    def test_fit(self):
        auto_est = AutoEstimator.from_torch(model_creator=model_creator,
                                            optimizer=get_optimizer,
                                            loss=nn.BCELoss(),
                                            logs_dir="/tmp/zoo_automl_logs",
                                            resources_per_trial={"cpu": 2},
                                            name="test_fit")
        data = get_train_val_data()
        auto_est.fit(data,
                     recipe=LinearRecipe(),
                     metric="accuracy")
        best_model = auto_est.get_best_model()
        assert best_model.optimizer.__class__.__name__ == "SGD"
        assert isinstance(best_model.loss_creator, nn.BCELoss)

    def test_fit_loss_name(self):
        auto_est = AutoEstimator.from_torch(model_creator=model_creator,
                                            optimizer=get_optimizer,
                                            loss="BCELoss",
                                            logs_dir="/tmp/zoo_automl_logs",
                                            resources_per_trial={"cpu": 2},
                                            name="test_fit")
        data = get_train_val_data()
        auto_est.fit(data,
                     recipe=LinearRecipe(),
                     metric="accuracy")
        best_model = auto_est.get_best_model()
        assert isinstance(best_model.loss_creator, nn.BCELoss)

    def test_fit_optimizer_name(self):
        auto_est = AutoEstimator.from_torch(model_creator=model_creator,
                                            optimizer="SGD",
                                            loss="BCELoss",
                                            logs_dir="/tmp/zoo_automl_logs",
                                            resources_per_trial={"cpu": 2},
                                            name="test_fit")
        data = get_train_val_data()
        auto_est.fit(data,
                     recipe=LinearRecipe(),
                     metric="accuracy")
        best_model = auto_est.get_best_model()
        assert best_model.optimizer.__class__.__name__ == "SGD"

    def test_fit_invalid_optimizer_name(self):
        invalid_optimizer_name = "ADAM"
        with pytest.raises(ValueError) as excinfo:
            auto_est = AutoEstimator.from_torch(model_creator=model_creator,
                                                optimizer=invalid_optimizer_name,
                                                loss="BCELoss",
                                                logs_dir="/tmp/zoo_automl_logs",
                                                resources_per_trial={"cpu": 2},
                                                name="test_fit")
        assert "valid torch optimizer name" in str(excinfo)

    def test_fit_invalid_loss_name(self):
        invalid_loss_name = "MAELoss"
        with pytest.raises(ValueError) as excinfo:
            auto_est = AutoEstimator.from_torch(model_creator=model_creator,
                                                optimizer="SGD",
                                                loss=invalid_loss_name,
                                                logs_dir="/tmp/zoo_automl_logs",
                                                resources_per_trial={"cpu": 2},
                                                name="test_fit")
        assert "valid torch loss name" in str(excinfo)

    def test_fit_multiple_times(self):
        auto_est = AutoEstimator.from_torch(model_creator=model_creator,
                                            optimizer="SGD",
                                            loss="BCELoss",
                                            logs_dir="/tmp/zoo_automl_logs",
                                            resources_per_trial={"cpu": 2},
                                            name="test_fit")
        data = get_train_val_data()
        auto_est.fit(data,
                     recipe=LinearRecipe(),
                     metric="accuracy")
        with pytest.raises(RuntimeError):
            auto_est.fit(data,
                         recipe=LinearRecipe(),
                         metric="accuracy")


if __name__ == "__main__":
    pytest.main([__file__])
