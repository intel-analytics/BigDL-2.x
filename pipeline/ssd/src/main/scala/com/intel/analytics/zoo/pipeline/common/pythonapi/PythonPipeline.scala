/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 *  limitations under the License.
 */

package com.intel.analytics.zoo.pipeline.common.pythonapi

import java.util
import java.util.{ArrayList, List => JList}

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.numeric._
import com.intel.analytics.bigdl.python.api.{JTensor, PythonBigDL}
import com.intel.analytics.bigdl.tensor.{Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.pipeline.common.dataset.roiimage.{FrcnnToBatch, ImageMiniBatch, RoiImageToBatch}
import com.intel.analytics.zoo.pipeline.common.{ModuleUtil, ObjectDetect}
import com.intel.analytics.zoo.transform.vision.pythonapi.{ImageFrame, PythonVisionTransform}
import org.apache.log4j.Logger
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.rdd.RDD

import scala.language.existentials
import scala.reflect.ClassTag

object PythonPipeline {

  def ofFloat(): PythonBigDL[Float] = new PythonPipeline[Float]()

  def ofDouble(): PythonBigDL[Double] = new PythonPipeline[Double]()

  val logger = Logger.getLogger(getClass)
}


class PythonPipeline[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonVisionTransform[T] {
  def shareMemory(model: Module[Float]): Module[Float] = {
    ModuleUtil.shareMemory(model)
    model
  }

  def objectPredict(model: Module[Float], imageBatchFrame: ImageBatchFrame): RDD[JTensor] = {
    val tensorRDD = ObjectDetect(imageBatchFrame.rdd, model)
    val listRDD = tensorRDD.map { res =>
      val tensor = res.asInstanceOf[Tensor[T]]
      toJTensor(tensor)
    }
    new JavaRDD[JTensor](listRDD)
  }

  def toSsdBatch(imageFrame: ImageFrame, batchSize: Int, nPartition: Int): ImageBatchFrame = {
    val toBatch = RoiImageToBatch(batchSize, false, Some(nPartition))
    ImageBatchFrame(toBatch(imageFrame.rdd).asInstanceOf[RDD[ImageMiniBatch]])
  }

  def toFrcnnBatch(imageFrame: ImageFrame, batchSize: Int, nPartition: Int): ImageBatchFrame = {
    val toBatch = FrcnnToBatch(batchSize, false, Some(nPartition))
    ImageBatchFrame(toBatch(imageFrame.rdd).asInstanceOf[RDD[ImageMiniBatch]])
  }
}

case class ImageBatchFrame(rdd: RDD[ImageMiniBatch]) {

}

