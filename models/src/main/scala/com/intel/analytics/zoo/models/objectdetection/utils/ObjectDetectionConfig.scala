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
    "ssd-vgg16-300x300-quantize",
    "ssd-vgg16-512x512",
    "ssd-vgg16-512x512-quantize",
    "ssd-mobilenet-300x300",
    "frcnn-vgg16",
    "frcnn-vgg16-compress",
    "frcnn-vgg16-quantize",
    "frcnn-vgg16-compress-quantize",
    "frcnn-pvanet",
    "frcnn-pvanet-quantize",
    "frcnn-pvanet-compress",
    "frcnn-vgg16-compress-quantize")

  def apply(model: String, dataset: String, version: String): Configure = {
    val labelMap = LabelReader(dataset)
    model match {
      case "ssd-vgg16-300x300" |
           "ssd-vgg16-300x300-quantize" =>
        Configure(ObjectDetectionConfig.preprocessSsdVgg(300, dataset, version),
          ScaleDetection(),
          batchPerPartition = 2,
          labelMap)
      case "ssd-vgg16-512x512" |
           "ssd-vgg16-512x512-quantize" =>
        Configure(ObjectDetectionConfig.preprocessSsdVgg(512, dataset, version),
          ScaleDetection(),
          batchPerPartition = 2,
          labelMap)
      case "ssd-mobilenet-300x300" =>
        Configure(ObjectDetectionConfig.preprocessSsdMobilenet(300, dataset, version),
          ScaleDetection(),
          batchPerPartition = 2,
          labelMap)
      case "frcnn-vgg16" |
           "frcnn-vgg16-quantize" |
           "frcnn-vgg16-compress" |
           "frcnn-vgg16-compress-quantize" =>
        Configure(ObjectDetectionConfig.preprocessFrcnnVgg(dataset, version),
          DecodeOutput(),
          batchPerPartition = 1,
          labelMap)
      case "frcnn-pvanet" |
           "frcnn-pvanet-quantize" |
           "frcnn-pvanet-compress" |
           "frcnn-pvanet-compress-quantize" =>
        Configure(ObjectDetectionConfig.preprocessFrcnnPvanet(dataset, version),
          DecodeOutput(),
          batchPerPartition = 1,
          labelMap)
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
      MatToTensor() -> ImInfo() -> ImageFrameToSample(Array(ImageFeature.imageTensor, "ImInfo"))
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

/**
 * Generate imInfo
 * imInfo is a tensor that contains height, width, scaleInHeight, scaleInWidth
 */
case class ImInfo() extends FeatureTransformer {
  override def transformMat(feature: ImageFeature): Unit = {
    feature("ImInfo") = feature.getImInfo()
  }
}
