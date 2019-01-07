/*
 * Copyright 2018 Analytics Zoo Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.zoo.pipeline.inference

import com.intel.analytics.zoo.pipeline.api.net.TFNet
import com.intel.analytics.zoo.pipeline.inference.DeviceType.DeviceTypeEnumVal

object InferenceModelFactory {

  def loadFloatInferenceModel(modelPath: String): FloatInferenceModel = {
    loadFloatInferenceModel(modelPath, null)
  }

  def loadFloatInferenceModel(modelPath: String, weightPath: String)
  : FloatInferenceModel = {
    val model = ModelLoader.loadFloatModel(modelPath, weightPath)
    model.evaluate()
    new FloatInferenceModel(model)
  }

  def loadFloatInferenceModelForCaffe(modelPath: String, weightPath: String)
  : FloatInferenceModel = {
    val model = ModelLoader.loadFloatModelForCaffe(modelPath, weightPath)
    model.evaluate()
    new FloatInferenceModel(model)
  }

  def loadFloatInferenceModelForTF(modelPath: String,
                                   intraOpParallelismThreads: Int = 1,
                                   interOpParallelismThreads: Int = 1,
                                   usePerSessionThreads: Boolean = true): FloatInferenceModel = {
    val sessionConfig = TFNet.SessionConfig(intraOpParallelismThreads,
      interOpParallelismThreads, usePerSessionThreads)
    val model = ModelLoader.loadFloatModelForTF(modelPath, sessionConfig)
    model.evaluate()
    new FloatInferenceModel(model)
  }

  def loadOpenvinoInferenceModelForTF(frozenModelFilePath: String,
                                      pipelineConfigFilePath: String,
                                      extensionsConfigFilePath: String,
                                      deviceType: DeviceTypeEnumVal): OpenVinoInferenceModel = {
    OpenVinoInferenceSupportive.loadTensorflowModel(
      frozenModelFilePath, pipelineConfigFilePath, extensionsConfigFilePath, deviceType)
  }

  def loadOpenvinoInferenceModelForIR(modelFilePath: String,
                     weightFilePath: String,
                     deviceType: DeviceTypeEnumVal): OpenVinoInferenceModel = {
    OpenVinoInferenceSupportive.loadOpenVinoIR(modelFilePath, weightFilePath, deviceType)
  }
}
