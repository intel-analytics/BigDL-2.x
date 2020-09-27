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
import numpy as np
from test.zoo.pipeline.utils.test_utils import ZooTestCase

from zoo.automl.config.recipe import LSTMGridRandomRecipe, MTNetGridRandomRecipe
from zoo.zouwu.autots.forecast import AutoTSTrainer
from zoo.zouwu.autots.forecast import TSPipeline

import pandas as pd


@pytest.mark.usefixtures("init_ray_context_fixture")
class TestZouwuAutoTS(ZooTestCase):

    def setup_method(self, method):
        # super(TestZouwuAutoTS, self).setup_method(method)
        self.create_data()

    def teardown_method(self, method):
        pass

    def create_data(self):
        sample_num = np.random.randint(100, 200)
        self.train_df = pd.DataFrame({"datetime": pd.date_range(
            '1/1/2019', periods=sample_num), "value": np.random.randn(sample_num)})
        val_sample_num = np.random.randint(20, 30)
        self.validation_df = pd.DataFrame({"datetime": pd.date_range(
            '1/1/2019', periods=val_sample_num), "value": np.random.randn(val_sample_num)})

    def test_AutoTSTrainer_smoke(self):
        horizon = np.random.randint(1, 6)
        tsp = AutoTSTrainer(dt_col="datetime",
                            target_col="value",
                            horizon=horizon,
                            extra_features_col=None
                            )
        pipeline = tsp.fit(self.train_df)
        assert isinstance(pipeline, TSPipeline)
        assert pipeline.internal.config is not None
        pipeline.evaluate(self.validation_df)
        pipeline.predict(self.validation_df)

    def test_AutoTrainer_LstmRecipe(self):
        horizon = np.random.randint(1, 6)
        tsp = AutoTSTrainer(dt_col="datetime",
                            target_col="value",
                            horizon=horizon,
                            extra_features_col=None
                            )
        pipeline = tsp.fit(self.train_df,
                           self.validation_df,
                           recipe=LSTMGridRandomRecipe(
                               num_rand_samples=5,
                               batch_size=[1024],
                               lstm_2_units=[8],
                               training_iteration=1,
                               epochs=1
                           ))
        assert isinstance(pipeline, TSPipeline)
        assert pipeline.internal.config is not None
        pipeline.evaluate(self.validation_df)
        pipeline.predict(self.validation_df)

    def test_AutoTrainer_MTNetRecipe(self):
        horizon = np.random.randint(1, 6)
        tsp = AutoTSTrainer(dt_col="datetime",
                            target_col="value",
                            horizon=horizon,
                            extra_features_col=None
                            )
        pipeline = tsp.fit(self.train_df,
                           self.validation_df,
                           recipe=MTNetGridRandomRecipe(
                               num_rand_samples=5,
                               time_step=[5],
                               long_num=[2],
                               batch_size=[1024],
                               cnn_hid_size=[32, 50],
                               training_iteration=1,
                               epochs=1
                           ))
        assert isinstance(pipeline, TSPipeline)
        assert pipeline.internal.config is not None
        pipeline.evaluate(self.validation_df)
        pipeline.predict(self.validation_df)


if __name__ == "__main__":
    pytest.main([__file__])
