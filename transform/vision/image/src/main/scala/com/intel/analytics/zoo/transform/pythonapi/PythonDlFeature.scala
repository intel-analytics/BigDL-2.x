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

package com.intel.analytics.zoo.transform.pythonapi

import java.util.{List => JList}

import com.intel.analytics.bigdl.numeric._
import com.intel.analytics.bigdl.python.api.PythonBigDL
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.transform.vision.image._
import com.intel.analytics.zoo.transform.vision.image.augmentation._
import com.intel.analytics.zoo.transform.vision.image.opencv.OpenCVMat
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.mllib.linalg.{DenseVector, Vectors}
import org.opencv.imgproc.Imgproc

import scala.collection.JavaConverters._
import scala.language.existentials
import scala.reflect.ClassTag

object PythonDlFeature {

  def ofFloat(): PythonBigDL[Float] = new PythonDlFeature[Float]()

  def ofDouble(): PythonBigDL[Double] = new PythonDlFeature[Double]()

}


class PythonDlFeature[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonBigDL[T] {

  def createTest(str: String): String = {
    return "hello " + str
  }

  def createResize(resizeH: Int, resizeW: Int, resizeMode: Int = Imgproc.INTER_LINEAR): Resize = {
    Resize(resizeH, resizeW, resizeMode)
  }

  def createColorJitter(): ColorJitter = {
    ColorJitter()
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

  def createCrop(): Crop = {
    Crop()
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

  def createRandomOp(transformer: FeatureTransformer, prob: Double): RandomTransformer = {
    RandomTransformer(transformer, prob)
  }

  def createSaturation(deltaLow: Double, deltaHigh: Double): Saturation = {
    Saturation(deltaLow, deltaHigh)
  }

  def transform(transformer: FeatureTransformer, data: JList[DenseVector]): JList[DenseVector] = {
    val pipeline = transformer -> MatToFloats()
    val start = System.nanoTime()
    val shape = data.get(1)
    val mat = OpenCVMat.floatToMat(data.get(0).toArray.map(_.toFloat),
      shape(0).toInt, shape(1).toInt)
    val feature = new ImageFeature()
    feature(ImageFeature.mat) = mat
    pipeline(feature)
    val denseVector = new DenseVector(feature.getFloats().map(_.toDouble))
    val transformedShape = Vectors.dense(feature.getHeight(), feature.getWidth(), 3).toDense
    println("transform takes " + (System.nanoTime() - start) / 1e9)
    List(denseVector, transformedShape).asJava
  }

  def transformRdd(transformer: FeatureTransformer, dataRdd: JavaRDD[JList[DenseVector]])
  : JavaRDD[JList[DenseVector]] = {
    val pipeline = transformer -> MatToFloats()
    val matRdd = dataRdd.rdd.map { tuple => {
      val shape = tuple.get(1)
      val data = tuple.get(0)
      // data.shape [h, w, c]
      val mat = OpenCVMat.floatToMat(data.toArray.map(_.toFloat),
        shape(0).toInt, shape(1).toInt)
      val feature = new ImageFeature()
      feature(ImageFeature.mat) = mat
      feature
    }
    }
    val transformed = pipeline.toIterator(matRdd)
    transformed.map(feature => {
      val denseVector = new DenseVector(feature.getFloats()
        .slice(0, feature.getHeight() * feature.getWidth() * 3).map(_.toDouble))
      val transformedShape = Vectors.dense(feature.getHeight(), feature.getWidth(), 3).toDense
      println(feature.getHeight(), feature.getWidth(), 3)
      List(denseVector, transformedShape).asJava
    })
  }

  def chainTransformer(list: JList[FeatureTransformer])
  : FeatureTransformer = {
    var cur = list.get(0)
    (1 until list.size()).foreach(t => cur = cur -> list.get(t))
    cur
  }

  def vector2Tensor(vector: DenseVector, shape: JList[Int]): Tensor[Float] = {
    val data = vector.toArray.map(_.toFloat)
    Tensor(Storage(data)).resize(shape.asScala.toArray)
  }
}



