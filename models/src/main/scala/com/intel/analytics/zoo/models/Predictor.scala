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

package com.intel.analytics.zoo.models

import scala.reflect.ClassTag
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.transform.vision.image._
import com.intel.analytics.zoo.models.objectdetection.utils.ObjectDetectionConfig

object Predictor {
  def predict[A <: Activity : ClassTag, B <: Activity : ClassTag, T: ClassTag]
  (model: AbstractModule[A, B, T],
    imageFrame: ImageFrame,
    outputLayer: String = null,
    shareBuffer: Boolean = false,
    predictKey: String = ImageFeature.predict): ImageFrame = {
    val config = Configure(model.getName())
    val result = imageFrame match {
      case distImageFrame: DistributedImageFrame =>
        model.predictImage(imageFrame -> config.preProcessor, outputLayer,
          shareBuffer, config.batchPerPartition, predictKey)
      case localImageFrame: LocalImageFrame =>
        throw new NotImplementedError("local predict is not supported for now, coming soon")
    }
    config.postProcessor(result)
  }
}

case class Configure(
  preProcessor: FeatureTransformer = null,
  postProcessor: FeatureTransformer = null,
  batchPerPartition: Int = 4) {
}

object Configure {

  val splitter = "_"

  /**
   * Get config for each model
   *
   * @param tag pub_model_dataset_version
   * @return
   */
  def apply(tag: String): Configure = {
    val splits = tag.split(splitter)
    require(splits.length >= 4, "tag needs at least 4 elements, publisher, model, dataset, version")
    require(splits(0) == "bigdl", "the model publisher needs to be BigDL")
    val model = splits(1)
    val dataset = splits(2)
    val version = splits(3)
    model.toLowerCase() match {
      case obModel if ObjectDetectionConfig.models contains obModel =>
        ObjectDetectionConfig(obModel, dataset, version)
    }
  }
}
