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

from test.zoo.pipeline.utils.test_utils import ZooTestCase
from zoo.automl.model.abstract import BaseModel
from zoo.automl.regression.time_sequence_predictor import *


class TestTimeSequencePredictor(ZooTestCase):

    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        pass

    def create_dataset(self):
        sample_num = np.random.randint(100, 200)
        train_df = pd.DataFrame({"datetime": pd.date_range(
            '1/1/2019', periods=sample_num), "value": np.random.randn(sample_num)})
        val_sample_num = np.random.randint(20, 30)
        validation_df = pd.DataFrame({"datetime": pd.date_range(
            '1/1/2019', periods=val_sample_num), "value": np.random.randn(val_sample_num)})
        future_seq_len = np.random.randint(1, 6)
        return train_df, validation_df, future_seq_len

    def test_fit_SmokeRecipe(self):
        train_df, validation_df, future_seq_len = self.create_dataset()
        tsp = TimeSequencePredictor(dt_col="datetime",
                                    target_col="value",
                                    future_seq_len=future_seq_len,
                                    extra_features_col=None, )
        pipeline = tsp.fit(train_df, validation_df)
        assert isinstance(pipeline, TimeSequencePipeline)
        assert isinstance(
            pipeline.feature_transformers,
            TimeSequenceFeatureTransformer)
        assert isinstance(pipeline.model, BaseModel)
        assert pipeline.config is not None

    def test_fit_LSTMGridRandomRecipe(self):
        train_df, _, future_seq_len = self.create_dataset()
        tsp = TimeSequencePredictor(dt_col="datetime",
                                    target_col="value",
                                    future_seq_len=future_seq_len,
                                    extra_features_col=None, )
        pipeline = tsp.fit(train_df,
                           recipe=LSTMGridRandomRecipe(
                               lstm_2_units=[4],
                               batch_size=[1024],
                               num_rand_samples=5,
                               look_back=2,
                               training_iteration=1,
                               epochs=1))
        assert isinstance(pipeline, TimeSequencePipeline)
        assert isinstance(
            pipeline.feature_transformers,
            TimeSequenceFeatureTransformer)
        assert isinstance(pipeline.model, BaseModel)
        assert pipeline.config is not None
        assert 'past_seq_len' in pipeline.config
        assert pipeline.config["past_seq_len"] == 2

    def test_fit_BayesRecipe(self):
        train_df, _, future_seq_len = self.create_dataset()
        tsp = TimeSequencePredictor(dt_col="datetime",
                                    target_col="value",
                                    future_seq_len=future_seq_len,
                                    extra_features_col=None, )
        pipeline = tsp.fit(
            train_df, recipe=BayesRecipe(
                num_samples=1,
                training_iteration=2,
                epochs=1,
                look_back=(3, 5)
            ))
        assert isinstance(pipeline, TimeSequencePipeline)
        assert isinstance(
            pipeline.feature_transformers,
            TimeSequenceFeatureTransformer)
        assert isinstance(pipeline.model, BaseModel)
        assert pipeline.config is not None
        assert "epochs" in pipeline.config
        assert [config_name for config_name in pipeline.config
                if config_name.startswith('bayes_feature')] == []
        assert [config_name for config_name in pipeline.config
                if config_name.endswith('float')] == []
        assert 'past_seq_len' in pipeline.config
        assert 3 <= pipeline.config["past_seq_len"] <= 5

    def test_fit_df_list(self):
        train_df, validation_df, future_seq_len = self.create_dataset()
        tsp = TimeSequencePredictor(dt_col="datetime",
                                    target_col="value",
                                    future_seq_len=future_seq_len,
                                    extra_features_col=None, )
        train_df_list = [train_df] * 3
        val_df_list = [validation_df] * 3
        pipeline = tsp.fit(train_df_list, val_df_list)
        assert isinstance(pipeline, TimeSequencePipeline)
        assert isinstance(
            pipeline.feature_transformers,
            TimeSequenceFeatureTransformer)
        assert isinstance(pipeline.model, BaseModel)
        assert pipeline.config is not None


if __name__ == '__main__':
    pytest.main([__file__])
