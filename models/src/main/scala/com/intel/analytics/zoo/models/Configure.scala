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

import com.intel.analytics.bigdl.dataset.PaddingParam
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image.FeatureTransformer
import com.intel.analytics.zoo.models.imageclassification.util.ImageClassificationConfig
import com.intel.analytics.zoo.models.objectdetection.utils.ObjectDetectionConfig

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
case class Configure[T: ClassTag](
  preProcessor: FeatureTransformer = null,
  postProcessor: FeatureTransformer = null,
  batchPerPartition: Int = 4,
  labelMap: Map[Int, String] = null,
  featurePaddingParam: Option[PaddingParam[T]] = None)(implicit ev: TensorNumeric[T]) {
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
  def parse[T: ClassTag](tag: String)(implicit ev: TensorNumeric[T]): Configure[T] = {
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
      case imcModel if ImageClassificationConfig.models contains imcModel =>
        ImageClassificationConfig(imcModel, dataset, version)
      case _ => throw new Exception(s"$model is not defined in BigDL model zoo")
    }
  }
}
