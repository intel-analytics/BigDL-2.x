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

package com.intel.analytics.zoo.models.image.imageclassification

import java.net.URL

import com.intel.analytics.bigdl.dataset.image.CropCenter
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image.ImageFeature
import com.intel.analytics.zoo.feature.common.Preprocessing
import com.intel.analytics.zoo.feature.image._
import com.intel.analytics.zoo.models.image.common.ImageConfigure

import scala.io.Source
import scala.reflect.ClassTag

object ImageClassificationConfig {
  val models = Set("alexnet",
    "alexnet-quantize",
    "inception-v1",
    "inception-v1-quantize",
    "inception-v3",
    "inception-v3-quantize",
    "resnet-50",
    "resnet-50-quantize",
    "resnet-50-int8",
    "vgg-16",
    "vgg-16-quantize",
    "vgg-19",
    "vgg-19-quantize",
    "densenet-161",
    "densenet-161-quantize",
    "squeezenet",
    "squeezenet-quantize",
    "mobilenet",
    "mobilenet-v2",
    "mobilenet-v2-quantize")

  def apply[T: ClassTag](model: String, dataset: String, version: String)
    (implicit ev: TensorNumeric[T]): ImageConfigure[T] = {
    dataset match {
      case "imagenet" => ImagenetConfig[T](model, dataset, version)
      case _ => throw new RuntimeException(s"dataset $dataset not supported for now")
    }
  }
}

object ImagenetConfig {

  val meanFile = getClass().getResource("/mean.txt")

  val mean : Array[Float] = createMean(meanFile)

  val imagenetLabelMap = LabelReader("IMAGENET")

  val imagenetInceptionV3LabelMap = LabelReader("IMAGENET", "inception-v3")

  def apply[T: ClassTag](model: String, dataset: String, version: String)
    (implicit ev: TensorNumeric[T]): ImageConfigure[T] = {
    model match {
      case "alexnet" |
            "alexnet-quantize" => ImageConfigure(preProcessor = alexnetPreprocessor,
        labelMap = imagenetLabelMap)
      case "inception-v1" |
           "inception-v1-quantize" => ImageConfigure(preProcessor = inceptionV1Preprocessor,
        labelMap = imagenetLabelMap)
      case "resnet-50" |
           "resnet-50-quantize" => ImageConfigure(preProcessor = resnetPreprocessor,
        labelMap = imagenetLabelMap)
      case "resnet-50-int8" => ImageConfigure(preProcessor = resnetNewPreprocessor,
        labelMap = imagenetLabelMap)
      case "vgg-16" |
           "vgg-16-quantize" => ImageConfigure(preProcessor = vggPreprocessor,
        labelMap = imagenetLabelMap)
      case "vgg-19" |
           "vgg-19-quantize" => ImageConfigure(preProcessor = vggPreprocessor,
        labelMap = imagenetLabelMap)
      case "densenet-161" |
           "densenet-161-quantize" => ImageConfigure(preProcessor = densenetPreprocessor,
        labelMap = imagenetLabelMap)
      case "squeezenet" |
           "squeezenet-quantize" => ImageConfigure(preProcessor = squeezenetPreprocessor,
        labelMap = imagenetLabelMap)
      case "mobilenet" |
           "mobilenet-v2" |
           "mobilenet-v2-quantize" => ImageConfigure(preProcessor = mobilenetPreprocessor,
        labelMap = imagenetLabelMap)
      case "inception-v3" |
           "inception-v3-quantize" => ImageConfigure(preProcessor = inceptionV3Preprocessor,
        labelMap = imagenetInceptionV3LabelMap)
    }
  }

  def alexnetPreprocessor() : Preprocessing[ImageFeature, ImageFeature] = {
    ImageResize(Consts.IMAGENET_RESIZE, Consts.IMAGENET_RESIZE) ->
      ImagePixelNormalizer(mean) -> ImageCenterCrop(227, 227) ->
      ImageMatToTensor() -> ImageSetToSample()
  }

  def commonPreprocessor(imageResizeSize: Int, imageCropSize : Int, meanR: Float, meanG: Float,
    meanB: Float, stdR: Float = 1, stdG: Float = 1, stdB: Float = 1):
    Preprocessing[ImageFeature, ImageFeature] = {
    ImageResize(imageResizeSize, imageResizeSize) ->
      ImageCenterCrop(imageCropSize, imageCropSize) -> ImageChannelNormalize(meanR, meanG, meanB,
      stdR, stdG, stdB) ->
      ImageMatToTensor() -> ImageSetToSample()
  }

  def inceptionV1Preprocessor(): Preprocessing[ImageFeature, ImageFeature] = {
    commonPreprocessor(Consts.IMAGENET_RESIZE, 224, 123, 117, 104)
  }

  def inceptionV3Preprocessor(): Preprocessing[ImageFeature, ImageFeature] = {
    commonPreprocessor(320, 299, 128, 128, 128, 128, 128, 128)
  }

  def resnetPreprocessor() : Preprocessing[ImageFeature, ImageFeature] = {
    commonPreprocessor(Consts.IMAGENET_RESIZE, 224, 123, 117, 104)
  }

  def resnetNewPreprocessor(): Preprocessing[ImageFeature, ImageFeature] = {
      ImageRandomResize(256, 256) ->
      ImageRandomCropper(224, 224, false, CropCenter) ->
      ImageChannelScaledNormalizer(104, 117, 123, 0.0078125) ->
      ImageMatToTensor() -> ImageSetToSample(targetKeys = Array("label"))
  }

  def vggPreprocessor(): Preprocessing[ImageFeature, ImageFeature] = {
    commonPreprocessor(Consts.IMAGENET_RESIZE, 224, 123, 117, 104)
  }

  def densenetPreprocessor() : Preprocessing[ImageFeature, ImageFeature] = {
    commonPreprocessor(Consts.IMAGENET_RESIZE, 224, 123, 117, 104, 1/0.017f, 1/0.017f, 1/0.017f)
  }

  def mobilenetPreprocessor() : Preprocessing[ImageFeature, ImageFeature] = {
    commonPreprocessor(Consts.IMAGENET_RESIZE, 224, 123.68f, 116.78f, 103.94f, 1/0.017f,
      1/0.017f, 1/0.017f )
  }

  def squeezenetPreprocessor(): Preprocessing[ImageFeature, ImageFeature] = {
    commonPreprocessor(Consts.IMAGENET_RESIZE, 227, 123, 117, 104)
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
