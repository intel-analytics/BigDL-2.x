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

from zoo.automl.model import BaseModel
from zoo.automl.regression.time_sequence_predictor import *


class TestTimeSequencePredictor:

    def test_fit(self):
        # sample_num should > past_seq_len, the default value of which is 50
        sample_num = 100
        dates = pd.date_range('1/1/2019', periods=sample_num)
        values = np.random.randn(sample_num)
        train_df = pd.DataFrame({"datetime": dates, "value": values})
        tsp = TimeSequencePredictor(dt_col="datetime",
                                    target_col="value",
                                    extra_features_col=None, )
        pipeline = tsp.fit(train_df)
        assert isinstance(pipeline, TimeSequencePipeline)
        assert isinstance(pipeline.feature_transformers, TimeSequenceFeatureTransformer)
        assert isinstance(pipeline.model, BaseModel)
        assert pipeline.config is not None


if __name__ == '__main__':
    pytest.main([__file__])

