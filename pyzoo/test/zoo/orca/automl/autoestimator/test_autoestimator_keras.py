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
import tensorflow as tf
import numpy as np
from unittest import TestCase
from zoo.orca.automl.auto_estimator import AutoEstimator
from zoo.automl.recipe.base import Recipe
import pytest


def model_creator(config):
    model = tf.keras.models.Sequential([tf.keras.layers.Dense(config["hidden_size"],
                                                              input_shape=(1,)),
                                        tf.keras.layers.Dense(1)])
    model.compile(loss="mse",
                  optimizer=tf.keras.optimizers.SGD(config["lr"]),
                  metrics=["mse"])
    return model


def get_train_val_data():
    def get_x_y(size):
        x = np.random.rand(size)
        y = x / 2

        x = x.reshape((-1, 1))
        y = y.reshape((-1, 1))
        return x, y
    train_x, train_y = get_x_y(size=1000)
    val_x, val_y = get_x_y(size=400)
    data = {'x': train_x, 'y': train_y, 'val_x': val_x, 'val_y': val_y}
    return data


class LinearRecipe(Recipe):
    def search_space(self, all_available_features):
        from zoo.orca.automl import hp
        return {
            "hidden_size": hp.choice([5, 10]),
            "lr": hp.choice([0.001, 0.003, 0.01]),
            "batch_size": hp.choice([32, 64])
        }

    def runtime_params(self):
        return {
            "training_iteration": 1,
            "num_samples": 4
        }


class TestTFKerasAutoEstimator(TestCase):
    def setUp(self) -> None:
        from zoo.orca import init_orca_context
        init_orca_context(cores=8, init_ray_on_spark=True)

    def tearDown(self) -> None:
        from zoo.orca import stop_orca_context
        stop_orca_context()

    def test_fit(self):
        auto_est = AutoEstimator.from_keras(model_creator=model_creator,
                                            logs_dir="/tmp/zoo_automl_logs",
                                            resources_per_trial={"cpu": 2},
                                            name="test_fit")
        data = get_train_val_data()
        auto_est.fit(data,
                     recipe=LinearRecipe(),
                     metric="mse")
        best_model = auto_est.get_best_model()
        assert "hidden_size" in best_model.config

    def test_fit_multiple_times(self):
        auto_est = AutoEstimator.from_keras(model_creator=model_creator,
                                            logs_dir="/tmp/zoo_automl_logs",
                                            resources_per_trial={"cpu": 2},
                                            name="test_fit")
        data = get_train_val_data()
        auto_est.fit(data,
                     recipe=LinearRecipe(),
                     metric="mse")
        with pytest.raises(RuntimeError):
            auto_est.fit(data,
                         recipe=LinearRecipe(),
                         metric="mse")


if __name__ == "__main__":
    pytest.main([__file__])
