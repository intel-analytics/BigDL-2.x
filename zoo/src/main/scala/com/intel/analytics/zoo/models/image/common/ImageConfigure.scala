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

package com.intel.analytics.zoo.models.image.common

import com.intel.analytics.bigdl.dataset.PaddingParam
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image.FeatureTransformer
import com.intel.analytics.zoo.models.image.objectdetection.ObjectDetectionConfig
import com.intel.analytics.zoo.models.image.imageclassification.ImageClassificationConfig

import org.apache.log4j.Logger

import scala.reflect.ClassTag

/**
 * predictor configure
 *
 * @param preProcessor preprocessor of ImageFrame before model inference
 * @param postProcessor postprocessor of ImageFrame after model inference
 * @param batchPerPartition batch size per partition
 * @param labelMap label mapping
 * @param featurePaddingParam featurePaddingParam if the inputs have variant size
 */
case class ImageConfigure[T: ClassTag](
  preProcessor: FeatureTransformer = null,
  postProcessor: FeatureTransformer = null,
  batchPerPartition: Int = 4,
  labelMap: Map[Int, String] = null,
  featurePaddingParam: Option[PaddingParam[T]] = None)(implicit ev: TensorNumeric[T]) {
}

object ImageConfigure {
  val logger = Logger.getLogger(getClass)

  val splitter = "_"

  /**
   * Get config for each model
   *
   * @param tag In 'publisher_model_dataset_version' format,
   * publisher is required to be bigdl in this model zoo
   * @return
   */
  def parse[T: ClassTag](tag: String)(implicit ev: TensorNumeric[T]): ImageConfigure[T] = {
    val splits = tag.split(splitter)
    require(splits.length >= 4, s"tag ${tag}" +
      s" needs at least 4 elements, publisher, model, dataset, version")
    require(splits(0) == "analytics-zoo", "the model publisher needs to be analytics-zoo")
    val model = splits(1)
    val dataset = splits(2)
    val version = splits(3)
    model.toLowerCase() match {
      case obModel if ObjectDetectionConfig.models contains obModel =>
        ObjectDetectionConfig(obModel, dataset, version)
      case imcModel if ImageClassificationConfig.models contains imcModel =>
        ImageClassificationConfig(imcModel, dataset, version)
      case _ => logger.warn(s"$model is not defined in Analytics zoo.")
        null
    }
  }
}
