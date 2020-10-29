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
from zoo.orca.data import SparkXShards
from zoo import get_node_and_core_number
from zoo.util import nest
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
        """
        Load an openVINO Estimator.

        :param model_path: String. The file path to the OpenVINO IR xml file.
        """
        return OpenvinoEstimatorWrapper(model_path=model_path)


class OpenvinoEstimatorWrapper(Estimator):
    def __init__(self,
                 *,
                 model_path):
        self.node_num, self.core_num = get_node_and_core_number()
        self.path = model_path
        self.model = InferenceModel(supported_concurrent_num=self.core_num)
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
            assert sc is not None, "You should pass sc(spark context) if data is a XShards."
            from zoo.orca.learn.tf.utils import convert_predict_to_xshard
            data = data.transform_shard(predict_transform)
            result_rdd = self.model.distributed_predict(data.rdd, sc)
            return convert_predict_to_xshard(result_rdd)
        elif isinstance(data, (np.ndarray, list)):
            if sc is None:
                return self.model.predict(data)
            else:
                total_core_num = self.core_num * self.node_num
                if isinstance(data, np.ndarray):
                    split_num = min(total_core_num, data.shape[0])
                    arrays = np.array_split(data, split_num)
                    data_rdd = sc.parallelize(arrays, numSlices=split_num)
                elif isinstance(data, list):
                    flattened = nest.flatten(data)
                    data_length = len(flattened[0])
                    data_to_be_rdd = []
                    split_num = min(total_core_num, flattened[0].shape[0])
                    for i in range(split_num):
                        data_to_be_rdd.append([])
                    for x in flattened:
                        assert len(x) == data_length, \
                            "the ndarrays in data must all have the same size in first dimension" \
                            ", got first ndarray of size {} and another {}".format(data_length,
                                                                                   len(x))
                        x_parts = np.array_split(x, split_num)
                        for idx, x_part in enumerate(x_parts):
                            data_to_be_rdd[idx].append(x_part)

                    data_to_be_rdd = [nest.pack_sequence_as(data, shard) for shard in
                                      data_to_be_rdd]
                    data_rdd = sc.parallelize(data_to_be_rdd, numSlices=split_num)

                result_rdd = self.model.distributed_predict(data_rdd, sc)
                result_arr_list = result_rdd.collect()
                result_arr = np.concatenate(result_arr_list, axis=0)
                return result_arr
        else:
            raise ValueError("Only XShards, a numpy array and a list of numpy arrays are supported "
                             "as input data, but get " + data.__class__.__name__)

    def evaluate(self, data, **kwargs):
        raise NotImplementedError

    def get_model(self):
        raise NotImplementedError

    def save(self, checkpoint):
        raise NotImplementedError

    def load(self, checkpoint):
        self.model.load_openvino(model_path=checkpoint,
                                 weight_path=checkpoint[:checkpoint.rindex(".")] + ".bin")
