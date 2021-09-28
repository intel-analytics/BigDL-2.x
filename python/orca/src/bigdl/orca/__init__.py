#
# Copyright 2016 The BigDL Authors.
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

from bigdl.dllib.utils.nncontext import *
from bigdl.dllib.utils.zoo_engine import prepare_env, is_spark_below_ver
from .common import OrcaContext, init_orca_context, stop_orca_context

prepare_env()
creator_classes = JavaCreator.get_creator_class()[:]
JavaCreator.set_creator_class([])
#JavaCreator.add_creator_class("com.intel.analytics.bigdl.dllib.utils.python.api.PythonBigDLKeras")
#JavaCreator.add_creator_class("com.intel.analytics.bigdl.dllib.utils.python.api.PythonBigDLOnnx")
#JavaCreator.add_creator_class("com.intel.analytics.bigdl.dllib.common.PythonZoo")
#JavaCreator.add_creator_class("com.intel.analytics.bigdl.dllib.nnframes.python.PythonNNFrames")
#JavaCreator.add_creator_class("com.intel.analytics.bigdl.dllib.feature.python.PythonImageFeature")
#JavaCreator.add_creator_class("com.intel.analytics.bigdl.dllib.feature.python.PythonTextFeature")
#JavaCreator.add_creator_class("com.intel.analytics.bigdl.dllib.feature.python.PythonFeatureSet")
#JavaCreator.add_creator_class("com.intel.analytics.bigdl.dllib.keras.python.PythonZooKeras")
#JavaCreator.add_creator_class("com.intel.analytics.bigdl.dllib.keras.python.PythonAutoGrad")
#JavaCreator.add_creator_class("com.intel.analytics.bigdl.dllib.estimator.python.PythonEstimator")
JavaCreator.add_creator_class("com.intel.analytics.bigdl.orca.tfpark.python.PythonTFPark")
JavaCreator.add_creator_class("com.intel.analytics.bigdl.orca.net.python.PythonZooNet")
JavaCreator.add_creator_class("com.intel.analytics.bigdl.orca.python.PythonOrca")
JavaCreator.add_creator_class("com.intel.analytics.bigdl.orca.inference.PythonInferenceModel")
#if not is_spark_below_ver("2.4"):
#    JavaCreator.add_creator_class("com.intel.analytics.zoo.friesian.python.PythonFriesian")
for clz in creator_classes:
    JavaCreator.add_creator_class(clz)

#__version__ = "0.12.0.dev0"
