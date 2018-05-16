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

import sys

from bigdl.nn.layer import Model as BModel
from bigdl.util.common import callBigDlFunc, to_list
from zoo.pipeline.api.keras.engine.topology import ZooKerasLayer, KerasNet

if sys.version >= '3':
    long = int
    unicode = str


class Net(BModel):

    def __init__(self, input, output, jvalue=None, bigdl_type="float", **kwargs):
        super(BModel, self).__init__(jvalue,
                                     to_list(input),
                                     to_list(output),
                                     bigdl_type,
                                     **kwargs)

    @staticmethod
    def from_jvalue(jvalue, bigdl_type="float"):
        """
        Create a Python Model base on the given java value
        :param jvalue: Java object create by Py4j
        :return: A Python Model
        """
        model = Net([], [], jvalue=jvalue, bigdl_type=bigdl_type)
        model.value = jvalue
        return model

    @staticmethod
    def load_bigdl(model_path, weight_path=None, bigdl_type="float"):
        """
        Load a pre-trained Bigdl model.

        :param path: The path containing the pre-trained model.
        :return: A pre-trained model.
        """
        jmodel = callBigDlFunc(bigdl_type, "netLoadBigDL", model_path, weight_path)
        return Net.from_jvalue(jmodel)

    @staticmethod
    def load(model_path, weight_path=None, bigdl_type="float"):
        jmodel = callBigDlFunc(bigdl_type, "netLoad", model_path, weight_path)
        return KerasNet.of(jmodel, bigdl_type)

    @staticmethod
    def load_torch(path, bigdl_type="float"):
        jmodel = callBigDlFunc(bigdl_type, "netLoadTorch", path)
        return Net.from_jvalue(jmodel, bigdl_type)

    @staticmethod
    def load_tf(path, inputs, outputs, byte_order="little_endian",
                bin_file=None, bigdl_type="float"):
        jmodel = callBigDlFunc(bigdl_type, "netLoadTF", path, inputs, outputs, byte_order, bin_file)
        return Net.from_jvalue(jmodel, bigdl_type)

    @staticmethod
    def load_caffe(def_path, model_path, bigdl_type="float"):
        jmodel = callBigDlFunc(bigdl_type, "netLoadCaffe", def_path, model_path)
        return Net.from_jvalue(jmodel, bigdl_type)

    def new_graph(self, outputs):
        value = callBigDlFunc(self.bigdl_type, "newGraph", self.value, outputs)
        return self.from_jvalue(value, self.bigdl_type)

    def freeze_up_to(self, names):
        callBigDlFunc(self.bigdl_type, "freezeUpTo", self.value, names)

    def unfreeze(self, names=None):
        callBigDlFunc(self.bigdl_type, "unFreeze", self.value, names)

    def toKeras(self):
        value = callBigDlFunc(self.bigdl_type, "netToKeras", self.value)
        return ZooKerasLayer.of(value, self.bigdl_type)
