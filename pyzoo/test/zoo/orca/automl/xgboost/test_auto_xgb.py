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
import pytest

from zoo.orca.automl.xgboost import AutoXGBRegressor

import numpy as np
from unittest import TestCase
from zoo.automl.recipe.base import Recipe


def get_data():
    def get_x_y(size):
        x = np.random.randn(size, 2)
        y = np.random.randn(size)
        return x, y
    train_x, train_y = get_x_y(1000)
    val_x, val_y = get_x_y(400)
    data = (train_x, train_y)
    validation_data = (val_x, val_y)
    return data, validation_data


def create_XGB_recipe():
    from zoo.orca.automl import hp
    return {
        "n_estimators": hp.randint(5, 10),
        "max_depth": hp.randint(2, 5),
        "lr": hp.loguniform(1e-4, 1e-1),
    }


class TestAutoXGBRegressor(TestCase):
    def setUp(self) -> None:
        from zoo.orca import init_orca_context
        init_orca_context(cores=8, init_ray_on_spark=True)

    def tearDown(self) -> None:
        from zoo.orca import stop_orca_context
        stop_orca_context()

    def test_fit(self):
        auto_xgb_reg = AutoXGBRegressor(cpus_per_trial=2,
                                        name="auto_xgb_regressor",
                                        tree_method='hist')
        data, validation_data = get_data()
        auto_xgb_reg.fit(data=data,
                         validation_data=validation_data,
                         search_space=create_XGB_recipe(),
                         n_sampling=4,
                         epochs=1,
                         metric="mae")
        best_model = auto_xgb_reg.get_best_model()
        assert 5 <= best_model.model.n_estimators <= 10
        assert 2 <= best_model.model.max_depth <= 5
        best_config = auto_xgb_reg.get_best_config()
        assert all(k in best_config.keys() for k in create_XGB_recipe().keys())

    def test_metric(self):
        auto_xgb_reg = AutoXGBRegressor(cpus_per_trial=2,
                                        name="auto_xgb_regressor",
                                        tree_method='hist')
        data, validation_data = get_data()
        with pytest.raises(ValueError) as exeinfo:
            auto_xgb_reg.fit(data=data,
                             epochs=1,
                             validation_data=validation_data,
                             metric="logloss",
                             search_space=create_XGB_recipe(),
                             n_sampling=4)
        assert "metric_mode" in str(exeinfo)
        auto_xgb_reg.fit(data=data,
                         epochs=1,
                         validation_data=validation_data,
                         metric="logloss",
                         metric_mode="min",
                         search_space=create_XGB_recipe(),
                         n_sampling=4)
