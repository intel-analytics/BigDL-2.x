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

package com.intel.analytics.zoo.pipeline.common

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.Transformer
import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.zoo.pipeline.common.dataset.roiimage.{RoiImageToBatch, SSDMiniBatch}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.ImageFeature
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

object Predictor {
  def predict(rdd: RDD[SSDMiniBatch],
    model: Module[Float],
    postProcess: (Tensor[Float], SSDMiniBatch) => Tensor[Float])
  : RDD[Tensor[Float]] = {
    model.evaluate()
    val broadcastModel = ModelBroadcast().broadcast(rdd.sparkContext, model)
    rdd.mapPartitions(dataIter => {
      val localModel = broadcastModel.value()
      dataIter.flatMap(batch => {
        var result = localModel.forward(batch.input).toTensor[Float]
        if (null != postProcess) result = postProcess(result, batch)
        if (result.dim() == 1) {
          Array(result)
        } else {
          result.split(1)
        }
      })
    })
  }

  def predict(rdd: RDD[ImageFeature],
    model: Module[Float], outputKey: String, toBatch: RoiImageToBatch,
    postProcess: (Tensor[Float], SSDMiniBatch) => Tensor[Float])
  : RDD[ImageFeature] = {
    require(toBatch.keepImageFeature)
    model.evaluate()
    val broadcastModel = ModelBroadcast().broadcast(rdd.sparkContext, model)
    val broadcastToBatch = rdd.sparkContext.broadcast(toBatch)
    rdd.mapPartitions(dataIter => {
      val localModel = broadcastModel.value()
      val localToBatch = broadcastToBatch.value.cloneTransformer()
      val miniBatch = localToBatch(dataIter)
      miniBatch.flatMap(batch => {
        var result = localModel.forward(batch.input).toTensor[Float]
        if (null != postProcess) result = postProcess(result, batch)
        val batchOut = if (result.dim() == 1) {
          Array(result)
        } else {
          result.split(1)
        }
        batch.imageFeatures.zip(batchOut).map(x => {
          x._1(outputKey) = x._2
          x._1
        })
      })
    })
  }
}

object Transform {
  def apply[A, B: ClassTag](rdd: RDD[A], transformer: Transformer[A, B]): RDD[B] = {
    val bcTransformer = rdd.sparkContext.broadcast(transformer)
    rdd.mapPartitions(dataIter => {
      val localTransformer = bcTransformer.value.cloneTransformer()
      localTransformer(dataIter)
    })
  }
}
