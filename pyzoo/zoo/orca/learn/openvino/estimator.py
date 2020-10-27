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
from zoo.pipeline.inference import InferenceModel
from zoo.orca.data import SparkXShards, SharedValue
import numpy as np
from bigdl.util.common import JTensor


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

    @staticmethod
    def from_openvino(*, model_path):
        return OpenvinoEstimatorWrapper(model_path=model_path)


class OpenvinoEstimatorWrapper(Estimator):
    def __init__(self,
                 *,
                 model_path):
        self.path = model_path
        self.model = InferenceModel()
        self.model.load_openvino(model_path=model_path,
                                 weight_path=model_path[:model_path.rindex(".")] + ".bin")

    def fit(self, data, epochs, **kwargs):
        raise NotImplementedError

    def predict(self, data, sc=None):
        def predict_transform(dict_data):
            assert isinstance(dict_data, dict), "each shard should be an dict"
            assert "x" in dict_data, "key x should in each shard"
            return dict_data["x"]

        if isinstance(data, SparkXShards):
            data = data.transform_shard(predict_transform)
            self.model.distributed_predict(data.rdd, sc)
        elif isinstance(data, (np.ndarray, list, JTensor)):
            return self.model.predict(data)
        else:
            raise ValueError("Only XShards, a numpy array, a list of numpy arrays, JTensor or "
                             "a list of JTensors are supported as input data, but get " +
                             data.__class__.__name__)

    def evaluate(self, data, **kwargs):
        pass

    def get_model(self):
        raise NotImplementedError

    def save(self, checkpoint):
        raise NotImplementedError

    def load(self, checkpoint):
        self.model.load_openvino(model_path=checkpoint,
                                 weight_path=checkpoint[:checkpoint.rindex(".")] + ".bin")
