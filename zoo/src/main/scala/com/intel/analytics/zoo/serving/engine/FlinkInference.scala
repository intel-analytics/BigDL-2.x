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

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.pipeline.inference.InferenceModel
import com.intel.analytics.zoo.serving.PreProcessing
import com.intel.analytics.zoo.serving.postprocessing.PostProcessing
import com.intel.analytics.zoo.serving.utils.{ClusterServingHelper, Conventions, SerParams}
import org.apache.flink.api.common.functions.RichMapFunction
import org.apache.flink.configuration.Configuration
import org.apache.log4j.Logger


class FlinkInference(params: SerParams)
  extends RichMapFunction[List[(String, String)], List[(String, String)]] {

  var t: Tensor[Float] = null
  var logger: Logger = null
  var pre: PreProcessing = null
  var post: PostProcessing = null

  override def open(parameters: Configuration): Unit = {
    logger = Logger.getLogger(getClass)

    if (ModelHolder.model == null) {
      ModelHolder.synchronized {
        if (ModelHolder.model == null) {
          val localModelDir = getRuntimeContext.getDistributedCache
            .getFile(Conventions.SERVING_MODEL_TMP_DIR).getPath
          val localConfPath = getRuntimeContext.getDistributedCache
            .getFile(Conventions.SERVING_CONF_TMP_PATH).getPath
          logger.info(s"Config parameters loaded at executor at path ${localConfPath}, " +
            s"Model loaded at executor at path ${localModelDir}")
          val helper = new ClusterServingHelper(localConfPath, localModelDir)
          helper.initArgs()
          ModelHolder.model = helper.loadInferenceModel()
        }
      }
    }


    pre = new PreProcessing(params)
  }

  override def map(in: List[(String, String)]): List[(String, String)] = {
    val t1 = System.nanoTime()
    val postProcessed = if (params.inferenceMode == "single") {
      val preProcessed = in.map(item => {
        val uri = item._1
        val input = pre.decodeArrowBase64(item._2)
        (uri, input)
      }).toIterator
      InferenceSupportive.singleThreadInference(preProcessed, params).toList
    } else {
      val preProcessed = in.grouped(params.coreNum).flatMap(itemBatch => {
        itemBatch.indices.toParArray.map(i => {
          val uri = itemBatch(i)._1
          val input = pre.decodeArrowBase64(itemBatch(i)._2)
          (uri, input)
        })
      })
      InferenceSupportive.multiThreadInference(preProcessed, params).toList
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
