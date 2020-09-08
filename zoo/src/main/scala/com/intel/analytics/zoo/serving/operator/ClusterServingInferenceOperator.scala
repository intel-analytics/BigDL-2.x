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

package com.intel.analytics.zoo.serving.operator

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.pipeline.inference.InferenceModel
import com.intel.analytics.zoo.serving.PreProcessing
import com.intel.analytics.zoo.serving.engine.ClusterServingInference
import com.intel.analytics.zoo.serving.postprocessing.PostProcessing
import com.intel.analytics.zoo.serving.utils.{ClusterServingHelper, Conventions, SerParams}
import org.apache.flink.api.common.functions.RichMapFunction
import org.apache.flink.configuration.Configuration
import org.apache.log4j.Logger


class ClusterServingInferenceOperator(params: ClusterServingParams)
  extends RichMapFunction[List[(String, String)], List[(String, String)]] {
  var logger: Logger = null
  var pre: PreProcessing = null

  override def open(parameters: Configuration): Unit = {
    logger = Logger.getLogger(getClass)

    if (ModelHolder.model == null) {
      ModelHolder.synchronized {
        if (ModelHolder.model == null) {
          logger.info("Loading Cluster Serving model...")
          ModelHolder.model = ClusterServingHelper
            .loadModelfromDir(params._modelPath, params._modelConcurrent)
        }
      }
    }


    pre = new PreProcessing(params)
  }

  override def map(in: List[(String, String)]): List[(String, String)] = {
    val t1 = System.nanoTime()
    val postProcessed = if (params._inferenceMode == "single") {
      val preProcessed = in.map(item => {
        val uri = item._1
        val input = pre.decodeArrowBase64(item._2)
        (uri, input)
      }).toIterator
      ClusterServingInference.singleThreadInference(preProcessed, params).toList
    } else {
      val preProcessed = in.grouped(params.coreNum).flatMap(itemBatch => {
        itemBatch.indices.toParArray.map(i => {
          val uri = itemBatch(i)._1
          val input = pre.decodeArrowBase64(itemBatch(i)._2)
          (uri, input)
        })
      })
      ClusterServingInference.multiThreadInference(preProcessed, params).toList
    }
    val t2 = System.nanoTime()
    logger.info(s"${postProcessed.size} records backend time ${(t2 - t1) / 1e9} s. " +
      s"Throughput ${postProcessed.size / ((t2 - t1) / 1e9)}")
    postProcessed
  }
}
object ModelHolder {
  var model: InferenceModel = null

}

