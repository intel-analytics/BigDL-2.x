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

from zoo.chronos.autots.model.auto_arima import AutoARIMA

import numpy as np
import pandas as pd
from unittest import TestCase
from zoo.automl.recipe.base import Recipe
from zoo.orca.automl import hp


def get_data():
    seq_len = 1095
    x = np.random.rand(seq_len)
    horizon = np.random.randint(2, 50)
    target = np.random.rand(horizon)
    data = {'x': x, 'y': None, 'val_x': None, 'val_y': target}
    return data


class TestAutoARIMA(TestCase):
    def setUp(self) -> None:
        from zoo.orca import init_orca_context
        init_orca_context(cores=8, init_ray_on_spark=True)

    def tearDown(self) -> None:
        from zoo.orca import stop_orca_context
        stop_orca_context()

    def test_fit(self):
        auto_arima = AutoARIMA()
        data = get_data()
        search_space = {
            "p": hp.randint(0, 4),
            "q": hp.randint(0, 5),
            "seasonality_mode": hp.choice([True, False]),
            "P": hp.randint(5, 12),
            "Q": hp.randint(0, 5),
            "m": hp.choice([4, 7]),
        }
        auto_arima.fit(data=data,
                       epochs=1,
                       metric="mse",
                       n_sampling=3,
                       search_space=search_space,
                       )
        best_model = auto_arima.get_best_model()
        assert 0 <= best_model.p <= 4
        assert 0 <= best_model.q <= 5
        assert 0 <= best_model.P <= 12
        assert 0 <= best_model.Q <= 5
        assert best_model.m in [4, 7]
