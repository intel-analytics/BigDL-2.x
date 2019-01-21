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

from bigdl.util.common import JavaValue, callBigDlFunc
from bigdl.nn.layer import Layer
from zoo.pipeline.api.keras.engine import KerasNet


class InferenceModel(JavaValue):
    def __init__(self, supported_concurrent_num=1, bigdl_type="float"):
        super(InferenceModel, self).__init__(None, bigdl_type, supported_concurrent_num)

    def load(self, model_path, weight_path=None):
        callBigDlFunc(self.bigdl_type, "inferenceModelLoad",
                      self.value, model_path, weight_path)

    def load_caffe(self, model_path, weight_path=None):
        callBigDlFunc(self.bigdl_type, "inferenceModelLoadCaffe",
                      self.value, model_path, weight_path)

    def load_openvino(self, model_path, weight_path):
        callBigDlFunc(self.bigdl_type, "inferenceModelLoadOpenVINO",
                      self.value, model_path, weight_path)

    def load_tf(self, model_path, backend="tensorflow",
                intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
                use_per_session_threads=True, model_type=None,
                pipeline_config_path=None, extensions_config_path=None):
        backend = backend.lower()
        if backend == "tensorflow" or backend == "tf":
            callBigDlFunc(self.bigdl_type, "inferenceModelTensorFlowLoadTF",
                          self.value, model_path, intra_op_parallelism_threads,
                          inter_op_parallelism_threads, use_per_session_threads)
        elif backend == "openvino" or backend == "ov":
            if model_type:
                callBigDlFunc(self.bigdl_type, "inferenceModelOpenVINOLoadTF",
                              self.value, model_path, model_type)
            else:
                assert pipeline_config_path is not None and extensions_config_path is not None,\
                    "For openvino backend, you must provide either model_type or both " \
                    "pipeline_config_path and extensions_config_path"
                callBigDlFunc(self.bigdl_type, "inferenceModelOpenVINOLoadTF",
                              self.value, model_path, pipeline_config_path, extensions_config_path)
        else:
            raise ValueError("Currently only tensorflow and openvino are supported as backend")

    def predict(self, inputs):
        jinputs, input_is_table = Layer.check_input(inputs)
        output = callBigDlFunc(self.bigdl_type,
                               "inferenceModelPredict",
                               self.value,
                               jinputs,
                               input_is_table)
        return KerasNet.convert_output(output)
