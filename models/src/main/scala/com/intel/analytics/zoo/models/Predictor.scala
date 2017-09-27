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

package com.intel.analytics.zoo.models

import com.intel.analytics.bigdl.dataset.Transformer
import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.nn.SoftMax
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.models.dataset.{ImageSample, PredictResult, SampleToBatch}
import com.intel.analytics.zoo.models.util.ImageNetLableReader
import org.apache.spark.rdd.RDD


trait Predictor {

  protected val batchPerPartition = 4

  var model : AbstractModule[Activity, Activity, Float] = null

  def predictLocal(path : String, topNum : Int,
                   preprocessor: Transformer[String, ImageSample] = null)
  : Array[PredictResult]

  def predictLocal(input : Tensor[Float], topNum : Int) : Array[PredictResult] = {
    val res = model.forward(input).asInstanceOf[Tensor[Float]]
    singleTopN(res, topNum)
  }

  def predictDistributed(paths : RDD[String], topNum : Int,
                         preprocessor: Transformer[String, ImageSample] = null):
    RDD[Array[PredictResult]]

  def predictDistributed(samples : RDD[ImageSample], topNum : Int):
   RDD[Array[PredictResult]] = {
    val partitionNum = samples.partitions.length
    val totalBatch = batchPerPartition * partitionNum
    val broadcastModel = ModelBroadcast[Float]().broadcast(samples.sparkContext, model)
    val broadcastProcessor = samples.sparkContext.broadcast(SampleToBatch(
      totalBatch = totalBatch, partitionNum = Some(partitionNum)))
    // val predictDataSet = paths.map(path => preprocessor(Seq(path).iterator))
    val predictResult = samples.mapPartitions(partition => {
      val localModel = broadcastModel.value()
      val localProcessor = broadcastProcessor.value.cloneTransformer()
      val minibatch = localProcessor(Seq(partition.next()).iterator)
      minibatch.map(batch => {
        val result = localModel.forward(batch.input).asInstanceOf[Tensor[Float]]
        topN(result, topNum)
      })
    })
    predictResult.flatMap(single => single)
  }

  protected def doPredictLocal(path : String, topNum : Int,
                               preprocessor: Transformer[String, ImageSample]) :  Array[PredictResult] = {
    val sample = preprocessor(Seq(path).iterator).next()
    val input = sample.input.view(Array(1) ++ sample.input.size())
    val res = model.forward(input).asInstanceOf[Tensor[Float]]
    singleTopN(res, topNum)
  }

  protected def doPredictDistributed(paths : RDD[String], topNum : Int ,
    preprocessor: Transformer[String, ImageSample]):
    RDD[Array[PredictResult]] = {
    val partitionNum = paths.partitions.length
    val totalBatch = batchPerPartition * partitionNum
    val broadcastModel = ModelBroadcast[Float]().broadcast(paths.sparkContext, model)
    val broadcastProcessor = paths.sparkContext.broadcast(SampleToBatch(
      totalBatch = totalBatch, partitionNum = Some(partitionNum)))
    preprocessor(paths)
    val predictDataSet = preprocessor(paths)
    val predictResult = predictDataSet.mapPartitions(partition => {
      val localModel = broadcastModel.value()
      val localProcessor = broadcastProcessor.value.cloneTransformer()
      val minibatch = localProcessor(partition)
      minibatch.map(batch => {
        val result = localModel.forward(batch.input).asInstanceOf[Tensor[Float]]
        topN(result, topNum)
      })
    })
    predictResult.flatMap(single => single)
  }

protected def topN(result : Tensor[Float], topN : Int) :
  Array[Array[PredictResult]] = {
  if (result.dim() == 1) {
    val single = singleTopN(result, topN)
    Array[Array[PredictResult]](single)
  } else {
    val total = result.size(1)
    var res =  new Array[Array[PredictResult]](total)
    var i = 0
    while (i < total) {
      val next = result.select(1, i + 1)
      val single = singleTopN(next, topN)
      res(i) = single
      i += 1
    }
    res
  }
}

protected def singleTopN(result : Tensor[Float], topN : Int) :
  Array[PredictResult] = {
    val total = result.nElement()
    val softMaxResult = SoftMax[Float]().forward(result)
    val sortedResult = softMaxResult.storage().array().zipWithIndex.sortWith(_._1 > _._1).toList.toArray
    val res = new Array[PredictResult](topN)
    var index = 0
    while (index < topN) {
      val className = ImageNetLableReader.labelByIndex(sortedResult(index)._2 + 1)
      val credict = sortedResult(index)._1
      res(index) = PredictResult(className, credict)
      index += 1
    }
    res
  }
}



