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

package com.intel.analytics.bigdl.pipeline.common.model

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.{SampleToMiniBatch}
import com.intel.analytics.bigdl.nn.SpatialShareConvolution
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.transform.vision.image._
import com.intel.analytics.zoo.pipeline.common.model.Preprocessor._
import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.utils.{Engine, MklBlas}
import com.intel.analytics.zoo.pipeline.common.dataset.roiimage.RoiImageToBatch
import com.intel.analytics.zoo.pipeline.common.model.Postprocessor._
import Model._

import scala.reflect.ClassTag

class Model[A <: Activity : ClassTag, B <: Activity : ClassTag, T: ClassTag]
(model: AbstractModule[A, B, T])(
  implicit ev: TensorNumeric[T]) {

  SpatialShareConvolution.shareConvolution(model)
  model.evaluate()

  // get tag
  val modelTag = "ssd_vgg16_300x300_pascal"
  val versionTag = "0.3.0"
  val defaultBatchPerPartition = 4

  private def defaultToBatch(totalBatch: Int, partitionNum: Int) = SampleToMiniBatch[T](
    batchSize = totalBatch,
    partitionNum = Some(partitionNum))

  private def defaultPostProcessor: (ImageFeature => ImageFeature) = null

  private def getResources(imf: ImageFrame, partitionNum: Int) = modelTag match {
    case "ssd_vgg16_300x300_pascal" =>
      val batchPerPartition = 2
      val totalBatch = partitionNum * batchPerPartition
      val toBatch = RoiImageToBatch(totalBatch, false, Some(partitionNum))
      val postProcessor = (imf: ImageFeature) => scaleDetection(imf)
      (preprocessSsdVgg300(imf), toBatch, postProcessor)

    case _ => throw new Exception(s"no model matches $modelTag")
  }

  val featureKey: String = "feature"



  def predictFeature(imageFrame: ImageFrame, shareBuffer: Boolean = false): ImageFrame = {
    imageFrame match {
      case distImageFrame: DistributedImageFrame =>
        val partitionNum = distImageFrame.rdd.partitions.length
        val (data, toBatch, postProcess) = getResources(distImageFrame, partitionNum)
        predictDistributed(model, data, toBatch, postProcess, shareBuffer)

      case localImageFrame: LocalImageFrame =>
        predictLocal(localImageFrame, shareBuffer)
    }
  }

}

object Model {
  implicit def abstractModuleToModel[T: ClassTag](model: AbstractModule[Activity, Activity, T])(
    implicit ev: TensorNumeric[T])
  : Model[Activity, Activity, T] = new Model[Activity, Activity, T](model)

  def predictDistributed[A <: ImageFeatureMiniBatch, T: ClassTag](
    model: AbstractModule[Activity, Activity, T],
    data: DistributedImageFrame,
    toBatch: ImageFeatureToBatch[A],
    postProcess: (ImageFeature => ImageFeature),
    shareBuffer: Boolean = false)(
    implicit ev: TensorNumeric[T])
  : ImageFrame = {
    val modelBroad = ModelBroadcast[T]()
      .broadcast(data.rdd.sparkContext, model.evaluate())
    val broadcastToBatch = data.rdd.sparkContext.broadcast(toBatch)
    val broadcastPostProcess = data.rdd.sparkContext.broadcast(postProcess)

    data.rdd.mapPartitions { partition =>
      val localModel = modelBroad.value()
      val localTransformer = broadcastToBatch.value.cloneTransformer()
      val localPostProcessor = broadcastPostProcess.value
      val miniBatch = localTransformer(partition)
      miniBatch.flatMap(batch => {
        val result = localModel.forward(batch.getInput()).toTensor[T]
        val batchOut = if (result.dim() == 1) {
          Array(result)
        } else {
          result.split(1)
        }
        batch.imageFeatures.zip(batchOut).map(x => {
          x._1(ImageFeature.feature) = x._2
          println(x._2)
          if (null != localPostProcessor) localPostProcessor(x._1)
          println(x._1(ImageFeature.feature))
          x._1
        })
      })
    }
  }

  def predictLocal(localImageFrame: LocalImageFrame, shareBuffer: Boolean = false): ImageFrame = {
    val subModelNumber = Engine.getEngineType match {
      case MklBlas => Engine.coreNumber()
      case _ => throw new IllegalArgumentException
    }
    throw new NotImplementedError()
  }
}
