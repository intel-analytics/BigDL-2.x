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

from zoo.automl.regression.time_sequence_predictor import TimeSequencePredictor
from zoo.automl.regression.time_sequence_predictor import SmokeRecipe
from zoo.automl.pipeline.time_sequence import Pipeline as ZooAutoMLPipeline


class AutoTrainer:

    def __init__(self, **config):
        self.internal = None
        self.set_config(**config)

    def set_config(self, **config):
        """
        set configurations for auto trainer
        """
        self.dt_col = config.get('dt_col', "datetime")
        self.target_col = config.get('target_col', "value")
        self.horizon = config.get('horizon', 1)
        self.extra_features_col = config.get('extra_features_col', None)

    def build(self):
        """
        Build the auto trainer
        """
        self.internal = TimeSequencePredictor(dt_col=self.dt_col,
                                              target_col=self.target_col,
                                              future_seq_len=self.horizon,
                                              extra_features_col=self.extra_features_col,
                                              )

    def fit(self,
            train_df,
            validation_df=None,
            metric="mse",
            recipe=SmokeRecipe(),
            mc=False,
            distributed=False,
            hdfs_url=None
            ):
        """
        fit w/ automl and return pipeline
        """
        zoo_pipeline = self.internal.fit(train_df,
                                         validation_df,
                                         metric,
                                         recipe,
                                         mc,
                                         distributed,
                                         hdfs_url)
        ppl = Pipeline()
        ppl.internal_ = zoo_pipeline
        return ppl


class Pipeline:
    """
    A wrapper of zoo automl pipeline
    """
    def __init__(self, pipeline_config=None):
        """
        Initializer not recommended to use.
        """
        if pipeline_config is None:
            self.internal_ = ZooAutoMLPipeline()
        else:
            self.internal_ = ZooAutoMLPipeline.load_from_config(pipeline_config)

    def fit(self, input_df,
            validation_df=None,
            mc=False,
            epoch_num=1):
        self.internal_.fit(input_df, validation_df, mc, epoch_num)

    def predict(self, input_df):
        return self.internal_.predict(input_df)

    def evaluate(self, input_df,
                    metrics=["mse"],
                    multioutput='raw_values'):
        return self.internal_.evaluate(input_df,metrics,multioutput)



if __name__ == '__main__':
    pass
