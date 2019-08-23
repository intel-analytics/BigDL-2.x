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
from zoo.automl.model import BaseModel
from zoo.automl.regression.time_sequence_predictor import *


class TestTimeSequencePredictor(ZooTestCase):

    def setup_method(self, method):
        super().setup_method(method)
        ray.init()

        # sample_num should > past_seq_len, the default value of which is 1
        sample_num = 100
        self.train_df = pd.DataFrame({"datetime": pd.date_range('1/1/2019',
                                                                periods=sample_num),
                                      "value": np.random.randn(sample_num)})
        val_sample_num = 16
        self.validation_df = pd.DataFrame({"datetime": pd.date_range('1/1/2019',
                                                                     periods=val_sample_num),
                                           "value": np.random.randn(val_sample_num)})

        self.future_seq_len_1 = 1
        self.tsp_1 = TimeSequencePredictor(dt_col="datetime",
                                           target_col="value",
                                           future_seq_len=self.future_seq_len_1,
                                           extra_features_col=None, )

        self.future_seq_len_3 = 3
        self.tsp_3 = TimeSequencePredictor(dt_col="datetime",
                                           target_col="value",
                                           future_seq_len=self.future_seq_len_3,
                                           extra_features_col=None, )
        self.default_past_seq_len = 1

    def teardown_method(self, method):
        """
        Teardown any state that was previously setup with a setup_method call.
        """
        super().teardown_method(method)
        ray.shutdown()

    def test_fit(self):
        pipeline_1 = self.tsp_1.fit(self.train_df, self.validation_df)
        pipeline_3 = self.tsp_3.fit(self.train_df, self.validation_df)
        assert isinstance(pipeline_1, TimeSequencePipeline)
        assert isinstance(pipeline_1.feature_transformers, TimeSequenceFeatureTransformer)
        assert isinstance(pipeline_1.model, BaseModel)
        assert pipeline_1.config is not None

    def test_fit_RandomRecipe(self):
        random_pipeline_1 = self.tsp_1.fit(self.train_df, self.validation_df,
                                           recipe=RandomRecipe(1))
        random_pipeline_3 = self.tsp_3.fit(self.train_df, self.validation_df,
                                           recipe=RandomRecipe(1))
        assert isinstance(random_pipeline_1, TimeSequencePipeline)
        assert isinstance(random_pipeline_1.feature_transformers,
                          TimeSequenceFeatureTransformer)
        assert isinstance(random_pipeline_1.model, BaseModel)
        assert random_pipeline_1.config is not None

    def test_fit_RandomRecipe_look_back(self):
        pipeline_look_back_tuple = self.tsp_1.fit(self.train_df, self.validation_df,
                                                  recipe=RandomRecipe(1, look_back=(2, 4)))
        assert 'past_seq_len' in pipeline_look_back_tuple.config
        assert 2 <= pipeline_look_back_tuple.config["past_seq_len"] <= 4

        pipeline_look_back_int = self.tsp_1.fit(self.train_df, self.validation_df,
                                                recipe=RandomRecipe(1, look_back=3))
        assert 'past_seq_len' in pipeline_look_back_int.config
        assert pipeline_look_back_int.config["past_seq_len"] == 3

    def test_fit_BayesRecipe(self):
        bayes_pipeline_1 = self.tsp_1.fit(self.train_df, self.validation_df,
                                          recipe=BayesRecipe(1))
        bayes_pipeline_3 = self.tsp_3.fit(self.train_df, self.validation_df,
                                          recipe=BayesRecipe(1))
        assert isinstance(bayes_pipeline_1, TimeSequencePipeline)
        assert isinstance(bayes_pipeline_1.feature_transformers,
                          TimeSequenceFeatureTransformer)
        assert isinstance(bayes_pipeline_1.model, BaseModel)
        assert bayes_pipeline_1.config is not None
        assert "epochs" in bayes_pipeline_1.config
        assert [config_name for config_name in bayes_pipeline_1.config
                if config_name.startswith('bayes_feature')] == []
        assert [config_name for config_name in bayes_pipeline_1.config
                if config_name.endswith('float')] == []

    def test_fit_BayesRecipe_look_back(self):
        pipeline_look_back_tuple = self.tsp_1.fit(self.train_df, self.validation_df,
                                                  recipe=BayesRecipe(1, look_back=(2, 4)))
        assert 'past_seq_len' in pipeline_look_back_tuple.config
        assert 2 <= pipeline_look_back_tuple.config["past_seq_len"] <= 4

        pipeline_look_back_int = self.tsp_1.fit(self.train_df, self.validation_df,
                                                recipe=BayesRecipe(1, look_back=3))
        assert 'past_seq_len' in pipeline_look_back_int.config
        assert pipeline_look_back_int.config["past_seq_len"] == 3

    def test_fit_df_list(self):
        train_df_list = [self.train_df]*3
        val_df_list = [self.validation_df]*3
        pipeline_1 = self.tsp_1.fit(train_df_list, val_df_list)
        pipeline_3 = self.tsp_3.fit(train_df_list, val_df_list)
        assert isinstance(pipeline_1, TimeSequencePipeline)
        assert isinstance(pipeline_1.feature_transformers, TimeSequenceFeatureTransformer)
        assert isinstance(pipeline_1.model, BaseModel)
        assert pipeline_1.config is not None


if __name__ == '__main__':
    pytest.main([__file__])
