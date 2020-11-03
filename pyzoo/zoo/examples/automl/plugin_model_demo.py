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
from zoo.automl.model import ModelBuilder
import torch
import torch.nn as nn
from zoo.automl.config.recipe import Recipe
from ray import tune
import pandas as pd
import numpy as np
from zoo.orca import init_orca_context


def model_creator(config):
    """Returns a torch.nn.Module object."""
    return nn.Linear(1, config.get("hidden_size", 1))


def optimizer_creator(model, config):
    """Returns optimizer defined upon the model parameters."""
    return torch.optim.SGD(model.parameters(), lr=config.get("lr", 1e-2))


def loss_creator(config):
    return nn.MSELoss()


class SimpleRecipe(Recipe):
    def __init__(self):
        super().__init__()
        self.num_samples = 2

    def search_space(self, all_available_features):
        return {
            "lr": tune.uniform(0.001, 0.01),
            "batch_size": tune.choice([32, 64])
        }


def get_data():
    def get_linear_data(a, b, size):
        x = np.arange(0, 10, 10 / size, dtype=np.float32)
        y = a*x + b
        return x, y
    train_x, train_y = get_linear_data(2, 5, 1000)
    df = pd.DataFrame({'x': train_x, 'y': train_y})
    val_x, val_y = get_linear_data(2, 5, 400)
    val_df = pd.DataFrame({'x': val_x, 'y': val_y})
    return df, val_df


if __name__ == "__main__":
    # 1. the way to enable auto tuning model from creators.
    init_orca_context(init_ray_on_spark=True)
    modelBuilder = ModelBuilder.from_pytorch(model_creator=model_creator,
                                             optimizer_creator=optimizer_creator,
                                             loss_creator=loss_creator)

    searcher = SearchEngineFactory.create_engine(backend="ray",
                                                 logs_dir="~/zoo_automl_logs",
                                                 resources_per_trial={"cpu": 2},
                                                 name="demo")

    # pass input data, modelbuilder and recipe into searcher.compile. Note that if user doesn't pass
    # feature transformer, the default identity feature transformer will be used.
    df, val_df = get_data()
    searcher.compile(df,
                     modelBuilder,
                     recipe=SimpleRecipe(),
                     feature_cols=["x"],
                     target_col="y",
                     validation_df=val_df)

    searcher.run()
    best_trials = searcher.get_best_trials(k=1)
    print(best_trials[0].config)

    # 2. you can also use the model builder with a fix config
    model = modelBuilder.build(config={
        "lr": 1e-2,  # used in optimizer_creator
        "batch_size": 32,  # used in data_creator
    })

    val_result = model.fit_eval(x=df[["x"]],
                                y=df[["y"]],
                                validation_data=(val_df[["x"]], val_df["y"]),
                                epochs=1)
    print(val_result)
