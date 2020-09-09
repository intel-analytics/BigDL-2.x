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

package com.intel.analytics.zoo.serving.engine

import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.zoo.pipeline.inference.InferenceModel
import com.intel.analytics.zoo.serving.postprocessing.PostProcessing
import org.apache.log4j.Logger

/**
 * Inference Logic of Cluster Serving
 * In Flink, ModelHolder will directly be used
 * model parameter is reserved for Spark backend
 */
object ClusterServingInference {
  val logger = Logger.getLogger(getClass)
  def singleThreadInference(preProcessed: Iterator[(String, Activity)],
                            modelType: String,
                            filterType: String = null,
                            model: InferenceModel = null): Iterator[(String, String)] = {
    val localModel = if (model == null) ModelHolder.model else model
    val postProcessed = preProcessed.map(pathByte => {
      try {
        val t = typeCheck(pathByte._2)
        dimCheck(t, "add", modelType)
        val result = localModel.doPredict(t)
        dimCheck(result, "remove", modelType)
        val value = PostProcessing(result.toTensor[Float], filterType, 1)
        (pathByte._1, value)
      } catch {
        case e: Exception =>
          logger.info(s"${e.printStackTrace()}, " +
            s"Your input ${pathByte._1} format is invalid to your model, this record is skipped")
          (pathByte._1, "NaN")
      }
    })
    postProcessed
  }
  def singleThreadBatchInference(preProcessed: Iterator[(String, Activity)],
                                 batchSize: Int,
                                 modelType: String,
                                 filterType: String = "",
                                 resizeFlag: Boolean = false,
                                 model: InferenceModel = null): Iterator[(String, String)] = {
    val localModel = if (model == null) ModelHolder.model else model
    val postProcessed = preProcessed.grouped(batchSize).flatMap(pathByte => {
      try {
        val thisBatchSize = pathByte.size
        val t = Timer.timing("batch", thisBatchSize) {
          batchInput(pathByte, batchSize, useMultiThreading = false, resizeFlag = resizeFlag)
        }
        dimCheck(t, "add", modelType)
        val result = Timer.timing("inference", thisBatchSize) {
          localModel.doPredict(t)
        }
        dimCheck(result, "remove", modelType)
        dimCheck(t, "remove", modelType)
        val kvResult = Timer.timing("postprocess", thisBatchSize) {
          (0 until thisBatchSize).toParArray.map(i => {
            val value = PostProcessing(result, filterType, i + 1)
            (pathByte(i)._1, value)
          })
        }
        kvResult
      } catch {
        case e: Exception =>
          logger.info(s"${e.printStackTrace()}, " +
            s"Your input format is invalid to your model, this batch is skipped")
          pathByte.toParArray.map(x => (x._1, "NaN"))
      }
    })
    postProcessed
  }
  def multiThreadInference(preProcessed: Iterator[(String, Activity)],
                           batchSize: Int,
                           modelType: String,
                           filterType: String = "",
                           resizeFlag: Boolean = false,
                           model: InferenceModel = null): Iterator[(String, String)] = {
    val localModel = if (model == null) ModelHolder.model else model
    val postProcessed = preProcessed.grouped(batchSize).flatMap(pathByteBatch => {
      try {
        val thisBatchSize = pathByteBatch.size
        val t = Timer.timing("batch", thisBatchSize) {
          batchInput(pathByteBatch, batchSize, resizeFlag)
        }

        /**
         * addSingletonDimension method will modify the
         * original Tensor, thus if reuse of Tensor is needed,
         * have to squeeze it back.
         */
        dimCheck(t, "add", modelType)
        val result = Timer.timing("inference", thisBatchSize) {
          localModel.doPredict(t)
        }
        dimCheck(result, "remove", modelType)
        dimCheck(t, "remove", modelType)
        val kvResult = Timer.timing("postprocess", thisBatchSize) {
          (0 until thisBatchSize).toParArray.map(i => {
            val value = PostProcessing(result, filterType, i + 1)
            (pathByteBatch(i)._1, value)
          })
        }
        kvResult
      } catch {
        case e: Exception =>
          logger.info(s"${e.printStackTrace()}, " +
            s"Your input format is invalid to your model, this batch is skipped")
          pathByteBatch.toParArray.map(x => (x._1, "NaN"))
      }
    })
    postProcessed
  }

  def batchInput(seq: Seq[(String, Activity)],
                 batchSize: Int,
                 useMultiThreading: Boolean,
                 resizeFlag: Boolean = true): Activity = {
    val thisBatchSize = seq.size
    println(s"This batch size is ${thisBatchSize.toString}")

    val inputSample = seq.head._2.toTable
    val kvTuples = inputSample.keySet.map(key => {
      (key, Tensor[Float](batchSize +:
        inputSample(key).asInstanceOf[Tensor[Float]].size()))
    }).toList
    val t = T(kvTuples.head, kvTuples.tail: _*)
    // Batch tensor and copy
    if (!useMultiThreading) {
      (0 until thisBatchSize).foreach(i => {
        val dataTable = seq(i)._2.toTable
        t.keySet.foreach(key => {
          t(key).asInstanceOf[Tensor[Float]].select(1, i + 1)
            .copy(dataTable(key).asInstanceOf[Tensor[Float]])
        })
      })
    } else {
      (0 until thisBatchSize).toParArray.foreach(i => {
        val dataTable = seq(i)._2.toTable
        t.keySet.foreach(key => {
          t(key).asInstanceOf[Tensor[Float]].select(1, i + 1)
            .copy(dataTable(key).asInstanceOf[Tensor[Float]])
        })
      })
    }
    // Resize and specific control
    if (resizeFlag) {
      t.keySet.foreach(key => {
        val singleTensorSize = inputSample(key).asInstanceOf[Tensor[Float]].size()
        var newSize = Array(thisBatchSize)
        for (elem <- singleTensorSize) {
          newSize = newSize :+ elem
        }
        t(key).asInstanceOf[Tensor[Float]].resize(newSize)
      })
    }
    if (t.keySet.size == 1) {
      t.keySet.foreach(key => {
        return t(key).asInstanceOf[Tensor[Float]]
      })
    }
    t
  }

  /**
   * Add or remove the singleton dimension for some specific model types
   * @param input the input to change dimension
   * @param op String, "add" or "remove"
   * @param modelType model type
   * @return input with dimension changed
   */
  def dimCheck(input: Activity, op: String, modelType: String): Activity = {
    if (modelType == "openvino") {
      if (input.isTensor) {
        op match {
          case "add" => input.asInstanceOf[Tensor[Float]].addSingletonDimension()
          case _ => input.asInstanceOf[Tensor[Float]].squeeze(1)
        }
      }
      else if (input.isTable) {
        val dataTable = input.toTable
        op match {
          case "add" => dataTable.keySet.foreach(key => {
            dataTable(key).asInstanceOf[Tensor[Float]].addSingletonDimension()
          })
          case _ => dataTable.keySet.foreach(key => {
            dataTable(key).asInstanceOf[Tensor[Float]].squeeze(1)
          })
        }
      }
    }
    input
  }

  /**
   * Use for single thread inference, to construct a batchSize = 1 input
   * Also return a Tensor if input Table has only one element
   * @param input Input table or tensor
   * @return input with single element batch constructed
   */
  def typeCheck(input: Activity): Activity = {
    if (input.isTable) {
      if (input.toTable.keySet.size == 1) {
        input.toTable.keySet.foreach(key => {
          return input.toTable(key).asInstanceOf[Tensor[Float]].addSingletonDimension()
        })
      }
      else {
        input.toTable.keySet.foreach(key => {
          input.toTable(key).asInstanceOf[Tensor[Float]].addSingletonDimension()
        })
      }
      input.toTable
    } else if (input.isTensor) {
      input.toTensor[Float].addSingletonDimension()
    } else {
      logger.error("Your input of Inference is neither Table nor Tensor, please check.")
      throw new Error("Your input is invalid, skipped.")
    }
  }
}
