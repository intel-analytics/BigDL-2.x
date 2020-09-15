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
from zoo.pipeline.nnframes import NNEstimator


class Estimator(object):
    def fit(self, data, epochs, **kwargs):
        pass

    def predict(self, data, **kwargs):
        pass

    def evaluate(self, data, **kwargs):
        pass

    def get_model(self):
        pass

    def save(self, checkpoint):
        pass

    def load(self, checkpoint):
        pass

    def shutdown(self, force=False):
        pass

    @staticmethod
    def from_bigdl(*, model, loss, feature_preprocessing=None, label_preprocessing=None,
                   input_type="spark_dataframe"):
        if input_type == "spark_dataframe":
            pass
        elif input_type == "featureset":
            pass
        else:
            raise ValueError("only horovod and bigdl backend are supported for now")


class NNEstimatorWrapper(Estimator):
    def __init__(self, *, model, optimizer, loss, feature_preprocessing=None,
                 label_preprocessing=None):
        self.estimator = NNEstimator(model, loss, feature_preprocessing,
                                     label_preprocessing).setOptimMethod(optimizer)

    def fit(self, data, epochs, feature_col, batch_size=32, caching_sample=True, val_data=None,
            val_trigger=None, val_methods=None, train_summary=None, val_summary=None,
            checkpoint_path=None, checkpoint_trigger=None):
        self.estimator.setBatchSize(batch_size).setMaxEpoch(epochs)\
            .setCachingSample(caching_sample).setFeatureCol(feature_col)
        if val_data is not None:
            pass
        if train_summary is not None:
            pass
        if val_summary is not None:
            pass
        if checkpoint_path is not None:
            pass

    def predict(self, data, **kwargs):
        pass

    def evaluate(self, data, **kwargs):
        pass

    def get_model(self):
        pass

    def save(self, checkpoint):
        pass

    def load(self, checkpoint):
        pass

    def shutdown(self, force=False):
        pass
