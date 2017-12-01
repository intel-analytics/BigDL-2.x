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

import java.util
import java.util.{List => JList}

import com.intel.analytics.bigdl.numeric._
import com.intel.analytics.bigdl.python.api.{JTensor, PythonBigDL, Sample}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.transform.vision.image._
import com.intel.analytics.zoo.transform.vision.image.augmentation._
import com.intel.analytics.zoo.transform.vision.image.opencv.OpenCVMat
import com.intel.analytics.zoo.transform.vision.label.roi._
import org.apache.log4j.Logger
import org.apache.spark.api.java.{JavaRDD, JavaSparkContext}
import org.apache.spark.sql.{SQLContext}
import org.opencv.imgproc.Imgproc

import collection.JavaConverters._
import scala.language.existentials
import scala.reflect.ClassTag

object PythonVisionTransform {

  def ofFloat(): PythonBigDL[Float] = new PythonVisionTransform[Float]()

  def ofDouble(): PythonBigDL[Double] = new PythonVisionTransform[Double]()

  val logger = Logger.getLogger(getClass)
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

  def createRandomCrop(cropWidth: Int, cropHeight: Int, isClip: Boolean): RandomCrop = {
    RandomCrop(cropWidth, cropHeight, isClip)
  }

  def createCenterCrop(cropWidth: Int, cropHeight: Int, isClip: Boolean): CenterCrop = {
    CenterCrop(cropWidth, cropHeight, isClip)
  }

  def createFixedCrop(wStart: Float, hStart: Float, wEnd: Float, hEnd: Float, normalized: Boolean,
    isClip: Boolean): FixedCrop = {
    FixedCrop(wStart, hStart, wEnd, hEnd, normalized, isClip)
  }

  def createDetectionCrop(roiKey: String, normalized: Boolean): DetectionCrop = {
    DetectionCrop(roiKey, normalized)
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

  def createChannelNormalize(meanR: Double, meanG: Double, meanB: Double,
    stdR: Double = 1, stdG: Double = 1, stdB: Double = 1): FeatureTransformer = {
    ChannelNormalize(meanR.toFloat, meanG.toFloat, meanB.toFloat,
      stdR.toFloat, stdG.toFloat, stdB.toFloat)
  }

  def createAspectScale(scale: Int, scaleMultipleOf: Int, maxSize: Int): FeatureTransformer = {
    AspectScale(scale, scaleMultipleOf, maxSize)
  }

  def createRoiCrop(): RoiCrop = {
    RoiCrop()
  }

  def createRoiExpand(): RoiExpand = {
    RoiExpand()
  }

  def createRoiHFlip(normalized: Boolean = true): RoiHFlip = {
    RoiHFlip(normalized)
  }

  def createRoiNormalize(): RoiNormalize = {
    RoiNormalize()
  }

  def transformImageFeature(transformer: FeatureTransformer, feature: ImageFeature)
  : ImageFeature = {
    transformer.transform(feature)
  }


  def transformImageFrame(transformer: FeatureTransformer,
    imageFrame: ImageFrame): ImageFrame = {
    imageFrame.transform(transformer)
  }

  def createDistributedImageFrame(imageRdd: JavaRDD[JTensor], labelRdd: JavaRDD[JTensor])
  : DistributedImageFrame = {
    require(null != imageRdd, "imageRdd cannot be null")
    val featureRdd = if (null != labelRdd) {
      imageRdd.rdd.zip(labelRdd.rdd).map(data => {
        createImageFeature(data._1, data._2)
      })
    } else {
      imageRdd.rdd.map(image => {
        createImageFeature(image, null)
      })
    }
    new DistributedImageFrame(featureRdd)
  }

  def createLocalImageFrame(images: JList[JTensor], labels: JList[JTensor])
  : LocalImageFrame = {
    require(null != images, "images cannot be null")
    val features = if (null != labels) {
      (0 until images.size()).map(i => {
        createImageFeature(images.get(i), labels.get(i))
      })
    } else {
      (0 until images.size()).map(i => {
        createImageFeature(images.get(i), null)
      })
    }
    new LocalImageFrame(features.toArray)
  }

  def createPipeline(list: JList[FeatureTransformer]): FeatureTransformer = {
    var cur = list.get(0)
    (1 until list.size()).foreach(t => cur = cur -> list.get(t))
    cur
  }


  def createImageFeature(data: JTensor = null, label: JTensor = null, uri: String = null)
  : ImageFeature = {
    val feature = new ImageFeature()
    if (null != data) {
      val mat = OpenCVMat.floatToMat(data.storage, data.shape(0), data.shape(1))
      feature(ImageFeature.mat) = mat
      feature(ImageFeature.size) = mat.shape()
    }
    if (null != label) {
      // todo: may need a method to change label format if needed
      feature(ImageFeature.label) = toTensor(label)
    }
    if (null != uri) {
      feature(ImageFeature.uri) = uri
    }
    feature
  }

  def createMatToFloats(validH: Int, validW: Int, validC: Int,
    meanR: Double = -1, meanG: Double = -1, meanB: Double = -1, outKey: String): MatToFloats = {
    val means = if (-1 != meanR) {
      Some(meanR.toFloat, meanG.toFloat, meanB.toFloat)
    } else None
    MatToFloats(validH, validW, validC, means, outKey)
  }

  def imageFeatureToSample(imageFeature: ImageFeature,
    floatKey: String = ImageFeature.floats, toChw: Boolean = true,
    withImInfo: Boolean = false): Sample = {
    val imageTensor = imageFeatureToImageTensor(imageFeature, floatKey, toChw)
    val features = new util.ArrayList[JTensor]()
    features.add(imageTensor)
    if (withImInfo) {
      val imInfo = imageFeature.getImInfo()
      features.add(toJTensor(imInfo.asInstanceOf[Tensor[T]]))
    }
    val label = imageFeatureToLabelTensor(imageFeature)
    Sample(features, label, "float")
  }

  def imageFeatureGetKeys(imageFeature: ImageFeature): JList[String] = {
    imageFeature.keys().toList.asJava
  }

  def distributedImageFrameToSampleRdd(imageFrame: DistributedImageFrame,
    floatKey: String = ImageFeature.floats, toChw: Boolean = true, withImInfo: Boolean = false)
  : JavaRDD[Sample] = {
    imageFrame.rdd.map(imageFeatureToSample(_, floatKey, toChw, withImInfo)).toJavaRDD()
  }

  def distributedImageFrameToImageTensorRdd(imageFrame: DistributedImageFrame,
    floatKey: String = ImageFeature.floats, toChw: Boolean = true): JavaRDD[JTensor] = {
    imageFrame.rdd.map(imageFeatureToImageTensor(_, floatKey, toChw)).toJavaRDD()
  }

  def distributedImageFrameToLabelTensorRdd(imageFrame: DistributedImageFrame): JavaRDD[JTensor] = {
    imageFrame.rdd.map(imageFeatureToLabelTensor).toJavaRDD()
  }

  def localImageFrameToSample(imageFrame: LocalImageFrame,
    floatKey: String = ImageFeature.floats, toChw: Boolean = true, withImInfo: Boolean = false)
  : JList[Sample] = {
    imageFrame.array.map(imageFeatureToSample(_, floatKey, toChw, withImInfo)).toList.asJava
  }

  def localImageFrameToImageTensor(imageFrame: LocalImageFrame,
    floatKey: String = ImageFeature.floats, toChw: Boolean = true): JList[JTensor] = {
    imageFrame.array.map(imageFeatureToImageTensor(_, floatKey, toChw)).toList.asJava
  }

  def localImageFrameToLabelTensor(imageFrame: LocalImageFrame): JList[JTensor] = {
    imageFrame.array.map(imageFeatureToLabelTensor).toList.asJava
  }

  def imageFeatureToImageTensor(imageFeature: ImageFeature,
    floatKey: String = ImageFeature.floats, toChw: Boolean = true): JTensor = {
    toJTensor(imageFeature.toTensor(floatKey, toChw).asInstanceOf[Tensor[T]])
  }

  def imageFeatureToLabelTensor(imageFeature: ImageFeature): JTensor = {
    val label = if (imageFeature.hasLabel()) {
      imageFeature.getLabel[Tensor[T]]
    } else {
      Tensor[T](1).fill(ev.fromType[Float](-1f))
    }
    toJTensor(label)
  }

  def read(path: String, sc: JavaSparkContext): ImageFrame = {
    if (sc == null) ImageFrame.read(path, null) else ImageFrame.read(path, sc.sc)
  }

  def readParquet(path: String, sc: SQLContext): DistributedImageFrame = {
    ImageFrame.readParquet(path, sc)
  }

  def isLocal(imageFrame: ImageFrame): Boolean = imageFrame.isLocal()

  def isDistributed(imageFrame: ImageFrame): Boolean = imageFrame.isDistributed()
}


