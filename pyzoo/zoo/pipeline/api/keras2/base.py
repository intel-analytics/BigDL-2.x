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

from bigdl.nn.layer import Layer
from bigdl.util.common import JavaValue

from zoo.pipeline.api.keras.base import ZooKerasLayer


class ZooKeras2Creator(JavaValue):
    def jvm_class_constructor(self):
        name = "createZooKeras2" + self.__class__.__name__
        print("creating: " + name)
        return name


class ZooKeras2Layer(ZooKeras2Creator, ZooKerasLayer):
    pass
