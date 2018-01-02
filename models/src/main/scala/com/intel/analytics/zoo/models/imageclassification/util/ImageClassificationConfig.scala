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

package com.intel.analytics.zoo.models.imageclassification.util

import java.net.URL

import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.transform.vision.image.{FeatureTransformer, ImageFrameToSample, MatToTensor}
import com.intel.analytics.bigdl.transform.vision.image.augmentation.{CenterCrop, ChannelNormalize, PixelNormalizer, Resize}
import com.intel.analytics.zoo.models.Configure

import scala.io.Source

object ImageClassificationConfig {
  val models = Set("alexnet",
    "inception-v1",
    "resnet-50",
    "vgg-16",
    "vgg-19",
    "densenet-161",
    "squeezenet",
    "mobilenet")

  def apply(model: String, dataset: String, version: String): Configure = {
    dataset match {
      case "imagenet" => ImagenetConfig(model, dataset, version)
      case _ => throw new RuntimeException(s"dataset $dataset not supported for now")
    }
  }
}

object ImagenetConfig {

  val meanFile = getClass().getResource("/mean.txt")

  val mean : Array[Float] = createMean(meanFile)

  val imagenetLabelMap = LabelReader("IMAGENET")

  def apply(model: String, dataset: String, version: String): Configure = {
    model match {
      case "alexnet" => Configure(preProcessor = alexnetPreprocessor,
        labelMap = imagenetLabelMap)
      case "inception-v1" => Configure(preProcessor = inceptionV1Preprocessor,
        labelMap = imagenetLabelMap)
      case "resnet-50" => Configure(preProcessor = resnetPreprocessor,
        labelMap = imagenetLabelMap)
      case "vgg-16" => Configure(preProcessor = vggPreprocessor,
        labelMap = imagenetLabelMap)
      case "vgg-19" => Configure(preProcessor = vggPreprocessor,
        labelMap = imagenetLabelMap)
      case "densenet-161" => Configure(preProcessor = densenetPreprocessor,
        labelMap = imagenetLabelMap)
      case "squeezenet" => Configure(preProcessor = squeezenetPreprocessor,
        labelMap = imagenetLabelMap)
      case "mobilenet" => Configure(preProcessor = mobilenetPreprocessor,
        labelMap = imagenetLabelMap)
    }
  }

  def alexnetPreprocessor() : FeatureTransformer = {
    Resize(Consts.IMAGENET_RESIZE, Consts.IMAGENET_RESIZE) ->
      PixelNormalizer(mean) -> CenterCrop(227, 227) ->
      MatToTensor() -> ImageFrameToSample()
  }

  def commonPreprocessor(imageSize : Int, meanR: Float, meanG: Float, meanB: Float,
                         stdR: Float = 1, stdG: Float = 1, stdB: Float = 1) : FeatureTransformer = {
    Resize(Consts.IMAGENET_RESIZE, Consts.IMAGENET_RESIZE) ->
      CenterCrop(imageSize, imageSize) -> ChannelNormalize(meanR, meanG, meanB,
      stdR, stdG, stdB) ->
      MatToTensor() -> ImageFrameToSample()
  }

  def inceptionV1Preprocessor(): FeatureTransformer = {
    commonPreprocessor(224, 123, 117, 104)
  }

  def resnetPreprocessor() : FeatureTransformer = {
    commonPreprocessor(224, 0.485f, 0.456f, 0.406f, 0.229f, 0.224f, 0.225f)
  }

  def vggPreprocessor(): FeatureTransformer = {
    commonPreprocessor(224, 123, 117, 104)
  }

  def densenetPreprocessor() : FeatureTransformer = {
    commonPreprocessor(224, 123, 117, 104, 1/0.017f, 1/0.017f, 1/0.017f)
  }

  def mobilenetPreprocessor() : FeatureTransformer = {
    commonPreprocessor(224, 123.68f, 116.78f, 103.94f, 1/0.017f, 1/0.017f, 1/0.017f )
  }

  def squeezenetPreprocessor(): FeatureTransformer = {
    commonPreprocessor(227, 123, 117, 104)
  }

  private def createMean(meanFile : URL) : Array[Float] = {
    val lines = Source.fromURL(meanFile).getLines.toArray
    val array = new Array[Float](lines.size)
    lines.zipWithIndex.foreach(data => {
      array(data._2) = data._1.toFloat
    })
    array
  }
}

sealed trait Dataset {
  val value: String
}

object Dataset {
  def apply(datasetString: String): Dataset = {
    datasetString.toUpperCase match {
      case Imagenet.value => Imagenet
    }
  }

  case object Imagenet extends Dataset {
    val value = "IMAGENET"
  }

}

object Consts {
  val IMAGENET_RESIZE : Int = 256
}
