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

import com.intel.analytics.bigdl.nn.SpatialShareConvolution
import com.intel.analytics.bigdl._
import scala.reflect.ClassTag
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image._
import com.intel.analytics.zoo.models.objectdetection.utils.ObjectDetectionConfig

class Predictor[T: ClassTag](
  model: AbstractModule[Activity, Activity, T],
  var configure: Configure = null
)(implicit ev: TensorNumeric[T]) {
  SpatialShareConvolution.shareConvolution[T](model)
  configure = if (null == configure) Configure.parse(model.getName()) else configure

  /**
   * Model prediction for BigDL model zoo.
   *
   * @param imageFrame local or distributed imageFrame
   * @param outputLayer output layer name, if it is null, use output of last layer
   * @param shareBuffer share buffer of output layer
   * @param predictKey key to store prediction result
   * @return imageFrame with prediction
   */
  def predict(imageFrame: ImageFrame,
    outputLayer: String = null,
    shareBuffer: Boolean = false,
    predictKey: String = ImageFeature.predict): ImageFrame = {

    // apply preprocessing if preProcessor is defined
    val data = if (null != configure.preProcessor) {
      imageFrame -> configure.preProcessor
    } else {
      imageFrame
    }

    val result = model.predictImage(data, outputLayer,
      shareBuffer, configure.batchPerPartition, predictKey)

    // apply post process if defined
    if (null != configure.postProcessor) configure.postProcessor(result) else result
  }
}

object Predictor {
  def apply[T: ClassTag](
    model: AbstractModule[Activity, Activity, T],
    configure: Configure = null)(implicit ev: TensorNumeric[T]): Predictor[T] =
    new Predictor(model, configure)
}


case class Configure(
  preProcessor: FeatureTransformer = null,
  postProcessor: FeatureTransformer = null,
  batchPerPartition: Int = 4,
  labelMap: Map[Int, String] = null) {
}

object Configure {

  val splitter = "_"

  /**
   * Get config for each model
   *
   * @param tag In 'publisher_model_dataset_version' format,
   * publisher is required to be bigdl in this model zoo
   * @return
   */
  def parse(tag: String): Configure = {
    val splits = tag.split(splitter)
    require(splits.length >= 4, s"tag ${tag}" +
      s" needs at least 4 elements, publisher, model, dataset, version")
    require(splits(0) == "bigdl", "the model publisher needs to be bigdl")
    val model = splits(1)
    val dataset = splits(2)
    val version = splits(3)
    model.toLowerCase() match {
      case obModel if ObjectDetectionConfig.models contains obModel =>
        ObjectDetectionConfig(obModel, dataset, version)
      case _ => throw new Exception(s"$model is not defined in BigDL model zoo")
    }
  }
}
