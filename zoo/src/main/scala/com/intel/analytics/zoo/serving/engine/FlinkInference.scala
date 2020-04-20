package com.intel.analytics.zoo.serving.engine

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.pipeline.inference.InferenceModel
import com.intel.analytics.zoo.serving.pipeline.{RedisIO, RedisUtils}
import com.intel.analytics.zoo.serving.utils.{ClusterServingHelper, SerParams}
import org.apache.flink.api.common.functions.RichMapFunction
import org.apache.flink.configuration.Configuration
import com.intel.analytics.zoo.serving.{PostProcessing, PreProcessing}
import com.intel.analytics.zoo.serving.ClusterServing.params
import org.apache.log4j.Logger


class FlinkInference(params: SerParams) extends RichMapFunction[List[(String, String)], List[(String, String)]] {
  var model: InferenceModel = null
  var t: Tensor[Float] = null
  var logger: Logger = null

  override def open(parameters: Configuration): Unit = {
    model = params.model
    logger = Logger.getLogger(getClass)
    t = if (params.chwFlag) {
      Tensor[Float](params.coreNum, params.C, params.H, params.W)
    } else {
      Tensor[Float](params.coreNum, params.H, params.W, params.C)
    }
  }

  override def map(in: List[(String, String)]): List[(String, String)] = {
    val t1 = System.nanoTime()
    val preProcessed = in.grouped(params.coreNum).flatMap(itemBatch => {
      itemBatch.indices.toParArray.map(i => {
        val uri = itemBatch(i)._1
        val tensor = PreProcessing(itemBatch(i)._2)
        (uri, tensor)
      })
    })

    val postProcessed = preProcessed.grouped(params.coreNum).flatMap(pathByteBatch => {
      val thisBatchSize = pathByteBatch.size



      (0 until thisBatchSize).toParArray.foreach(i =>
        t.select(1, i + 1).copy(pathByteBatch(i)._2))

      val thisT = if (params.chwFlag) {
        t.resize(thisBatchSize, params.C, params.H, params.W)
      } else {
        t.resize(thisBatchSize, params.H, params.W, params.C)
      }
      val x = if (params.modelType == "openvino") {
        thisT.addSingletonDimension()
      } else {
        thisT
      }
      /**
       * addSingletonDimension method will modify the
       * original Tensor, thus if reuse of Tensor is needed,
       * have to squeeze it back.
       */
      val result = if (params.modelType == "openvino") {
        val res = model.doPredict(x).toTensor[Float].squeeze()
        t.squeeze(1)
        res
      } else {
        model.doPredict(x).toTensor[Float]
      }
      (0 until thisBatchSize).toParArray.map(i => {
        val value = PostProcessing(result.select(1, i + 1), params.filter)
        (pathByteBatch(i)._1, value)
      })
    }).toList

    val t2 = System.nanoTime()
    logger.info(s"${postProcessed.size} records backend time ${(t2 - t1) / 1e9} s")
    postProcessed
  }
}
