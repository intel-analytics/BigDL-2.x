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

package com.intel.analytics.zoo.models.objectdetection.utils

import com.intel.analytics.bigdl.transform.vision.image.augmentation.{AspectScale, ChannelNormalize, Resize}
import com.intel.analytics.bigdl.transform.vision.image._
import com.intel.analytics.zoo.models.objectdetection.utils.Dataset.{Coco, Pascal}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.zoo.models.Configure

object ObjectDetectionConfig {

  val models = Set("ssd-vgg16-300x300",
    "ssd-vgg16-512x512",
    "ssd-mobilenet-300x300",
    "frcnn-vgg16",
    "frcnn-vgg16-compress",
    "frcnn-pvanet",
    "frcnn-pvanet-compress",
    "ssd-vgg16-300x300-quantize")

  def apply(model: String, dataset: String, version: String): Configure = {
    model match {
      case "ssd-vgg16-300x300" |
           "ssd-vgg16-300x300-quantize" =>
        Configure(ObjectDetectionConfig.preprocessSsdVgg(300, dataset, version),
          ScaleDetection(),
          batchPerPartition = 2)
      case "ssd-vgg16-512x512" =>
        Configure(ObjectDetectionConfig.preprocessSsdVgg(512, dataset, version),
          ScaleDetection(),
          batchPerPartition = 2)
      case "ssd-mobilenet-300x300" =>
        Configure(ObjectDetectionConfig.preprocessSsdMobilenet(300, dataset, version),
          ScaleDetection(),
          batchPerPartition = 2)
      case "frcnn-vgg16" =>
        Configure(ObjectDetectionConfig.preprocessFrcnnVgg(dataset, version),
          batchPerPartition = 1)
      case "frcnn-pvanet" =>
        Configure(ObjectDetectionConfig.preprocessFrcnnPvanet(dataset, version),
          batchPerPartition = 1)
    }
  }

  def preprocessSsdVgg(resolution: Int, dataset: String, version: String)
  : FeatureTransformer = {
    preprocessSsd(resolution, (123f, 117f, 104f), 1f)
  }

  def preprocessSsd(resolution: Int, meansRGB: (Float, Float, Float),
    scale: Float): FeatureTransformer = {
    Resize(resolution, resolution) ->
      ChannelNormalize(meansRGB._1, meansRGB._2, meansRGB._3, scale, scale, scale) ->
      MatToTensor() -> ImageFrameToSample()
  }

  def preprocessSsdMobilenet(resolution: Int, dataset: String, version: String)
  : FeatureTransformer = {
    Dataset(dataset) match {
      case Pascal =>
        preprocessSsd(resolution, (127.5f, 127.5f, 127.5f), 1 / 0.007843f)
      case Coco =>
        throw new Exception("coco is not yet supported for BigDL ssd mobilenet")
    }
  }

  def preprocessFrcnn(resolution: Int, scaleMultipleOf: Int): FeatureTransformer = {
    AspectScale(resolution, scaleMultipleOf) ->
      ChannelNormalize(122.7717f, 115.9465f, 102.9801f) ->
      MatToTensor() -> ImageFrameToSample()
  }

  def preprocessFrcnnVgg(dataset: String, version: String): FeatureTransformer = {
    Dataset(dataset) match {
      case Pascal =>
        preprocessFrcnn(600, 1)
      case Coco =>
        throw new Exception("coco is not yet supported for BigDL FrcnnVgg")
    }
  }

  def preprocessFrcnnPvanet(dataset: String, version: String): FeatureTransformer = {
    Dataset(dataset) match {
      case Pascal =>
        preprocessFrcnn(640, 32)
      case Coco =>
        throw new Exception("coco is not yet supported for BigDL FrcnnPvanet")
    }
  }
}

sealed trait Dataset {
  val value: String
}

object Dataset {
  def apply(datasetString: String): Dataset = {
    datasetString.toUpperCase match {
      case Pascal.value => Pascal
      case Coco.value => Coco
    }
  }

  case object Pascal extends Dataset {
    val value = "PASCAL"
  }

  case object Coco extends Dataset {
    val value: String = "COCO"
  }

}
