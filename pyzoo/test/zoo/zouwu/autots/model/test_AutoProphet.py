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

from zoo.zouwu.autots.model.AutoProphet import AutoProphet

import numpy as np
import pandas as pd
from unittest import TestCase
from zoo.automl.recipe.base import Recipe
from zoo.orca.automl import hp


def get_data():
    seq_len = 480
    x = pd.DataFrame(pd.date_range('20130101', periods=seq_len), columns=['ds'])
    x.insert(1, 'y', np.random.rand(seq_len))
    horizon = np.random.randint(2, 50)
    target = pd.DataFrame(pd.date_range('20140426', periods=horizon), columns=['ds'])
    target.insert(1, 'y', np.random.rand(horizon))
    data = {'x': x, 'y': None, 'val_x': None, 'val_y': target}
    return data


class ProphetRecipe(Recipe):
    def __init__(self):
        super().__init__()
        self.training_iteration = 1  # period of report metric in fit_eval
        self.num_samples = 10  # sample hyperparameters

    def search_space(self):
        return {
            "changepoint_prior_scale": hp.loguniform(0.001, 0.5),
            "seasonality_prior_scale": hp.loguniform(0.01, 10),
            "holidays_prior_scale": hp.loguniform(0.01, 10),
            "seasonality_mode": hp.choice(['additive', 'multiplicative']),
            "changepoint_range": hp.uniform(0.8, 0.95)
        }

    def runtime_params(self):
        return {
            "training_iteration": self.training_iteration,
            "num_samples": self.num_samples
        }


class TestAutoProphet(TestCase):
    def setUp(self) -> None:
        from zoo.orca import init_orca_context
        init_orca_context(cores=8, init_ray_on_spark=True)

    def tearDown(self) -> None:
        from zoo.orca import stop_orca_context
        stop_orca_context()

    def test_fit(self):
        auto_prophet = AutoProphet()
        data = get_data()
        auto_prophet.fit(data=data,
                         recipe=ProphetRecipe(),
                         metric="mse")
        best_model = auto_prophet.get_best_model()
        assert 0.001 <= best_model.model.changepoint_prior_scale <= 0.5
        assert 0.01 <= best_model.model.seasonality_prior_scale <= 10
        assert 0.01 <= best_model.holidays_prior_scale <= 10
        assert best_model.model.seasonality_mode in ['additive', 'multiplicative']
        assert 0.8 <= best_model.model.changepoint_range <= 0.95
