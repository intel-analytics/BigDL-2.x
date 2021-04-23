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

from zoo.automl.search import SearchEngineFactory
from zoo.automl.model import PytorchModelBuilder, KerasModelBuilder
import torch
import tensorflow as tf
import torch.nn as nn
from zoo.automl.recipe.base import Recipe
from zoo.orca.automl import hp
import numpy as np
from zoo.orca import init_orca_context


def model_creator(config):
    """Returns a torch.nn.Module object."""
    return nn.Linear(1, config.get("hidden_size", 1))


def model_creator_keras(config):
    """Returns a tf.keras model"""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1)
    ])
    model.compile(loss="mse",
                  optimizer='sgd',
                  metrics=["mse"])
    return model


def optimizer_creator(model, config):
    """Returns optimizer defined upon the model parameters."""
    return torch.optim.SGD(model.parameters(), lr=config.get("lr", 1e-2))


def loss_creator(config):
    return nn.MSELoss()


class SimpleRecipe(Recipe):
    def __init__(self):
        super().__init__()
        self.num_samples = 2
        self.training_iteration = 20

    def search_space(self, all_available_features):
        return {
            "lr": hp.uniform(0.01, 0.02),
            "batch_size": hp.choice([16, 32, 64])
        }


def get_data():
    def get_linear_data(a, b, size):
        x = np.arange(0, 10, 10 / size, dtype=np.float32)
        y = a*x + b
        return x, y
    train_x, train_y = get_linear_data(2, 5, 1000)
    val_x, val_y = get_linear_data(2, 5, 400)
    data = {'x': train_x, 'y': train_y, 'val_x': val_x, 'val_y': val_y}
    return data


if __name__ == "__main__":
    # 1. the way to enable auto tuning model from creators.
    init_orca_context(init_ray_on_spark=True)
    modelBuilder = PytorchModelBuilder(model_creator=model_creator,
                                       optimizer_creator=optimizer_creator,
                                       loss_creator=loss_creator)

    searcher = SearchEngineFactory.create_engine(backend="ray",
                                                 logs_dir="~/zoo_automl_logs",
                                                 resources_per_trial={"cpu": 2},
                                                 name="demo")

    # pass input data, modelbuilder and recipe into searcher.compile. Note that if user doesn't pass
    # feature transformer, the default identity feature transformer will be used.
    data = get_data()
    searcher.compile(data=data,
                     model_create_func=modelBuilder,
                     recipe=SimpleRecipe())

    searcher.run()
    best_trials = searcher.get_best_trials(k=1)
    print(best_trials[0].config)

    # rebuild this best config model and evaluate
    best_model = modelBuilder.build_from_ckpt(best_trials[0].model_path)
    searched_best_model = best_model.evaluate(data["val_x"], data["val_y"], metrics=["rmse"])

    # 2. you can also use the model builder with a fix config
    model = modelBuilder.build(config={
        "lr": 1e-2,  # used in optimizer_creator
        "batch_size": 32,  # used in data_creator
    })

    model.fit_eval(x=data["x"],
                   y=data["y"],
                   validation_data=(data["val_x"], data["val_y"]),
                   epochs=20)
    val_result_pytorch_manual = model.evaluate(x=data["x"], y=data["y"], metrics=['rmse'])

    # 3. try another modelbuilder based on tfkeras
    modelBuilder_keras = KerasModelBuilder(model_creator_keras)
    model = modelBuilder_keras.build(config={
        "lr": 1e-2,  # used in optimizer_creator
        "batch_size": 32,  # used in data_creator
        "metric": "mse"
    })

    model.fit_eval(x=data["x"],
                   y=data["y"],
                   validation_data=(data["val_x"], data["val_y"]),
                   epochs=20)
    val_result_tensorflow_manual = model.evaluate(x=data["x"], y=data["y"], metrics=['rmse'])
    print("Searched best model validation rmse:", searched_best_model)
    print("Pytorch model validation rmse:", val_result_pytorch_manual)
    print("Tensorflow model validation rmse:", val_result_tensorflow_manual)
