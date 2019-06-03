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
import os

from zoo.automl.pipeline.abstract import Pipeline
from zoo.automl.common.util import save_config, load_config
from zoo.automl.feature.time_sequence import TimeSequenceFeatureTransformer
from zoo.automl.model import VanillaLSTM


class TimeSequencePipeline(Pipeline):

    def __init__(self, feature_transformers=None, model=None, config=None):
        """
        initialize a pipeline
        :param model: the internal model
        :param feature_transformers: the feature transformers
        """
        if feature_transformers is None:
            assert model is None and config is None
            self.feature_transformers = TimeSequenceFeatureTransformer()
            self.model = VanillaLSTM(check_optional_config=False)
            print("Initialize new time sequence pipeline.")
        else:
            self.feature_transformers = feature_transformers
            self.model = model
            self.config = config

    def evaluate(self,
                 input_df,
                 metric=["mean_squared_error"]):
        """
        evaluate the pipeline
        :param input_df:
        :param metric:
        :return:
        """
        x, y = self.feature_transformers.transform(input_df, is_train=True)
        return self.model.evaluate(x, y, metric)

    def predict(self, input_df):
        # there might be no y in the data, TODO needs to fix in TimeSquenceFeatures
        x = self.feature_transformers.transform(input_df, is_train=False)
        y_pred = self.model.predict(x)
        y_output = self.feature_transformers.post_processing(y_pred)
        return y_output

    def save(self, file):
        """
        save pipeline to file, contains feature transformer, model, trial config.
        :param file:
        :return:
        """
        if not os.path.isdir(file):
            os.mkdir(file)
        model_path = os.path.join(file, "weights_tune.h5")
        config_path = os.path.join(file, "all_config.json")
        self.feature_transformers.save(config_path, replace=True)
        self.model.save(model_path, config_path)
        # check if ** is needed
        save_config(config_path, self.config)

    def restore(self, file):
        """
        restore pipeline from file
        :param file:
        :param config:
        :return:
        """       
        model_path = os.path.join(file, "weights_tune.h5")
        config_path = os.path.join(file, "all_config.json")
        all_config = load_config(config_path)
        self.model.restore(model_path, **all_config)
        self.feature_transformers.restore(**all_config)
