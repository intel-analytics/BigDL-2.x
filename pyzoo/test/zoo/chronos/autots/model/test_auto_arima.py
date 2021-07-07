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
    np.random.seed(0)
    seq_len = 400
    data = np.random.rand(seq_len)
    horizon = np.random.randint(2, 50)
    validation_data = np.random.rand(horizon)
    return data, validation_data


class TestAutoARIMA(TestCase):
    def setUp(self) -> None:
        from zoo.orca import init_orca_context
        init_orca_context(cores=8, init_ray_on_spark=True)

    def tearDown(self) -> None:
        from zoo.orca import stop_orca_context
        stop_orca_context()

    def test_fit(self):
        data, validation_data = get_data()
        auto_arima = AutoARIMA(metric="mse",
                               p=hp.randint(0, 4),
                               q=hp.randint(0, 4),
                               seasonality_mode=hp.choice([True, False]),
                               P=hp.randint(5, 12),
                               Q=hp.randint(5, 12),
                               m=hp.choice([4, 7])
                               )
        auto_arima.fit(data=data,
                       validation_data=validation_data,
                       epochs=1,
                       n_sampling=1,
                       )
        best_model = auto_arima.get_best_model()
