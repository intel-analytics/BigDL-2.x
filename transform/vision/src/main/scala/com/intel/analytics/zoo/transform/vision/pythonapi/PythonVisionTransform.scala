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

package com.intel.analytics.zoo.transform.vision.pythonapi

import java.util.{List => JList}

import com.intel.analytics.bigdl.numeric._
import com.intel.analytics.bigdl.dataset.{Sample => JSample}
import com.intel.analytics.bigdl.python.api.{JTensor, PythonBigDL, Sample}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.transform.vision.image._
import com.intel.analytics.zoo.transform.vision.image.augmentation._
import com.intel.analytics.zoo.transform.vision.image.opencv.OpenCVMat
import com.intel.analytics.zoo.transform.vision.label.roi._
import com.intel.analytics.zoo.transform.vision.util.NormalizedBox
import org.apache.spark.api.java.JavaRDD
import org.opencv.imgproc.Imgproc

import scala.language.existentials
import scala.reflect.ClassTag

object PythonVisionTransform {

  def ofFloat(): PythonBigDL[Float] = new PythonVisionTransform[Float]()

  def ofDouble(): PythonBigDL[Double] = new PythonVisionTransform[Double]()

}


class PythonVisionTransform[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonBigDL[T] {

  def createTest(str: String): String = {
    return "hello " + str
  }

  def createResize(resizeH: Int, resizeW: Int, resizeMode: Int = Imgproc.INTER_LINEAR): Resize = {
    Resize(resizeH, resizeW, resizeMode)
  }

  def createColorJitter(brightnessProb: Double = 0.5, brightnessDelta: Double = 32,
    contrastProb: Double = 0.5, contrastLower: Double = 0.5, contrastUpper: Double = 1.5,
    hueProb: Double = 0.5, hueDelta: Double = 18,
    saturationProb: Double = 0.5, saturationLower: Double = 0.5, saturationUpper: Double = 1.5,
    randomOrderProb: Double = 0, shuffle: Boolean = false): ColorJitter = {
    ColorJitter(brightnessProb, brightnessDelta, contrastProb,
      contrastLower, contrastUpper, hueProb, hueDelta, saturationProb,
      saturationLower, saturationUpper, randomOrderProb, shuffle)
  }

  def createBrightness(deltaLow: Double, deltaHigh: Double): Brightness = {
    Brightness(deltaLow, deltaHigh)
  }

  def createChannelOrder(): ChannelOrder = {
    ChannelOrder()
  }

  def createContrast(deltaLow: Double, deltaHigh: Double): Contrast = {
    Contrast(deltaLow, deltaHigh)
  }

  def createRandomCrop(cropWidth: Int, cropHeight: Int): RandomCrop = {
    RandomCrop(cropWidth, cropHeight)
  }

  def createCenterCrop(cropWidth: Int, cropHeight: Int): CenterCrop = {
    CenterCrop(cropWidth, cropHeight)
  }

  def createCrop(normalized: Boolean = true, roi: JList[Double], roiKey: String)
  : Crop = {
    if (roi != null) {
      Crop(normalized, bbox = Some(NormalizedBox(roi.get(0).toFloat, roi.get(1).toFloat,
        roi.get(2).toFloat, roi.get(3).toFloat)))
    } else if (!roiKey.isEmpty) {
      Crop(normalized, roiKey = Some(roiKey))
    } else {
      null
    }
  }

  def createExpand(meansR: Int = 123, meansG: Int = 117, meansB: Int = 104,
    maxExpandRatio: Double = 4.0): Expand = {
    Expand(meansR, meansG, meansB, maxExpandRatio)
  }

  def createHFlip(): HFlip = {
    HFlip()
  }

  def createHue(deltaLow: Double, deltaHigh: Double): Hue = {
    Hue(deltaLow, deltaHigh)
  }

  def createRandomTransformer(transformer: FeatureTransformer, prob: Double): RandomTransformer = {
    RandomTransformer(transformer, prob)
  }

  def createSaturation(deltaLow: Double, deltaHigh: Double): Saturation = {
    Saturation(deltaLow, deltaHigh)
  }

  def createRandomSampler(): FeatureTransformer = {
    RandomSampler()
  }

  def createChannelNormalize(meanR: Int, meanG: Int, meanB: Int): FeatureTransformer = {
    ChannelNormalize((meanR, meanG, meanB))
  }

  def createRoiCrop(): RoiCrop = {
    RoiCrop()
  }

  def createRoiExpand(): RoiExpand = {
    RoiExpand()
  }

  def createRoiHFlip(): RoiHFlip = {
    RoiHFlip()
  }

  def createRoiNormalize(): RoiNormalize = {
    RoiNormalize()
  }

  def transformImageFeature(transformer: FeatureTransformer, feature: ImageFeature)
  : ImageFeature = {
    transformer.transform(feature)
  }

  def transformImageFeatureRdd(transformer: FeatureTransformer, dataRdd: ImageFeatureRdd)
  : ImageFeatureRdd = {
    ImageFeatureRdd(transformer(dataRdd.rdd))
  }

  def tensorRddToImageFeatureRdd(dataRdd: JavaRDD[JList[JTensor]]): ImageFeatureRdd = {
    val featureRdd = dataRdd.rdd.map(data => {
      var image: JTensor = null
      var label: JTensor = null
      if (data.size() > 0) {
        image = data.get(0)
      }
      if (data.size() > 1) {
        label = data.get(1)
      }
      createImageFeature(image, label)
    })
    ImageFeatureRdd(featureRdd)
  }

  def chainFeatureTransformer(list: JList[FeatureTransformer])
  : FeatureTransformer = {
    var cur = list.get(0)
    (1 until list.size()).foreach(t => cur = cur -> list.get(t))
    cur
  }


  def createImageFeature(data: JTensor = null, label: JTensor = null, path: String = null)
  : ImageFeature = {
    val feature = new ImageFeature()
    if (null != data) {
      feature(ImageFeature.mat) = OpenCVMat.floatToMat(data.storage, data.shape(0), data.shape(1))
    }
    if (null != label) {
      // todo: may need a method to change label format if needed
      feature(ImageFeature.label) = toTensor(label)
    }
    if (null != path) {
      feature(ImageFeature.path) = path
    }
    feature
  }

  def createMatToFloats(validH: Int, validW: Int,
    meanR: Int = -1, meanG: Int = -1, meanB: Int = -1, outKey: String): MatToFloats = {
    val means = if (-1 != meanR) {
      Some(meanR, meanG, meanB)
    } else None
    MatToFloats(validH, validW, means, outKey)
  }

  def imageFeatureToSample(imageFeature: ImageFeature): Sample = {
    require(imageFeature.contains(ImageFeature.floats),
      "there is no float array in image feature, please call MatToFloats().transform first")
    // transpose the shape of image from (h, w, c) to (c, h, w)
    val image = Tensor(Storage(imageFeature.getFloats()))
      .resize(imageFeature.getHeight(), imageFeature.getWidth(), 3)
      .transpose(1, 3).transpose(2, 3).contiguous()
    val label = if (imageFeature.hasLabel()) {
      imageFeature.getLabel[Tensor[Float]]
    } else {
      Tensor[Float](1).fill(-1)
    }
    val jSample = JSample[Float](image, label)
    toPySample(jSample.asInstanceOf[JSample[T]])
  }

  def imageFeatureRddToSampleRdd(imageFeatureRdd: ImageFeatureRdd): JavaRDD[Sample] = {
    imageFeatureRdd.rdd.map(imageFeatureToSample).toJavaRDD()
  }

  def getImage(imageFeature: ImageFeature): JTensor = {
    if (imageFeature.contains(ImageFeature.floats)) {
      JTensor(imageFeature.getFloats(), Array(imageFeature.getHeight(), imageFeature.getWidth(), 3),
        "float")
    } else {
      val mat = imageFeature.opencvMat()
      val floats = new Array[Float](mat.height() * mat.width() * 3)
      OpenCVMat.toFloatBuf(mat, floats)
      JTensor(floats, Array(mat.height(), mat.width(), 3), "float")
    }
  }
}



