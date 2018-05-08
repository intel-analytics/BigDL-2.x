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

from bigdl.nn.layer import Container, Layer
from bigdl.util.common import JavaValue, callBigDlFunc

if sys.version >= '3':
    long = int
    unicode = str


class ZooModelCreator(JavaValue):
    def jvm_class_constructor(self):
        name = "createZoo" + self.__class__.__name__
        print("creating: " + name)
        return name


class ZooModel(ZooModelCreator, Container):
    """
    The base class for models in Analytics Zoo.
    """
    def save_model(self, path, weight_path=None, over_write=False):
        """
        Save the model to the specified path.

        # Arguments
        path: The path to save the model. Local file system, HDFS and Amazon S3 are supported.
              HDFS path should be like 'hdfs://[host]:[port]/xxx'.
              Amazon S3 path should be like 's3a://bucket/xxx'.
        weight_path: The path to save weights. Default is None.
        over_write: Whether to overwrite the file if it already exists. Default is False.
        """
        callBigDlFunc(self.bigdl_type, "saveZooModel",
                      self.value, path, weight_path, over_write)

    @staticmethod
    def _do_load(jmodel, bigdl_type="float"):
        model = Layer(jvalue=jmodel, bigdl_type=bigdl_type)
        model.value = jmodel
        return model
