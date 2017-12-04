/*
 * Copyright 2016 The BigDL Authors.
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

package com.intel.analytics.zoo.models.squeezenet

import com.intel.analytics.bigdl.dataset.Transformer
import com.intel.analytics.bigdl.utils.serializer.ModuleLoader
import com.intel.analytics.zoo.models.Predictor
import com.intel.analytics.zoo.models.dataset.{ImageSample, ImageToBytes, MateToSample, PredictResult}
import com.intel.analytics.zoo.transform.vision.image.{BytesToMat}
import com.intel.analytics.zoo.transform.vision.image.augmentation.{CenterCrop, ChannelNormalize, Resize}
import org.apache.spark.rdd.RDD

@SerialVersionUID(- 2010415206552736051L)
class SqueezenetPredictor(modelPath: String) extends Predictor with Serializable {
  model = ModuleLoader.
    loadFromFile[Float](modelPath).evaluate()


  val transformer = ImageToBytes() -> BytesToMat() -> Resize(256, 256) ->
    CenterCrop(227, 227) ->
    ChannelNormalize(123f, 117f, 104f) ->
    MateToSample(false)

  override def predictLocal(path : String, topNum : Int,
                            preprocessor: Transformer[String, ImageSample] = transformer)
  : PredictResult = {
    doPredictLocal(path, topNum, preprocessor)
  }

  override def predictDistributed(paths : RDD[String], topNum : Int,
                                  preprocessor: Transformer[String, ImageSample] = transformer):
  RDD[PredictResult] = {
    doPredictDistributed(paths, topNum, preprocessor)
  }
}

object SqueezenetPredictor {
  def apply(modelPath: String): SqueezenetPredictor = new SqueezenetPredictor(modelPath)
}

