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

import java.util.concurrent.TimeUnit
import java.util.concurrent.locks.ReentrantLock

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

  var logger: Logger = null
  var inference: ClusterServingInference = null

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
    inference = new ClusterServingInference(new PreProcessing(
      params.chwFlag, params.redisHost, params.redisPort),
      params.modelType, params.filter, params.coreNum, params.resize)
  }

  override def map(in: List[(String, String)]): List[(String, String)] = {
    val t1 = System.nanoTime()
    val postProcessed = {
      if (params.modelType == "openvino") {
        inference.multiThreadPipeline(in)
      } else {
        inference.singleThreadPipeline(in)
      }
    }

    val t2 = System.nanoTime()
    logger.info(s"${in.size} records backend time ${(t2 - t1) / 1e9} s. " +
      s"Throughput ${in.size / ((t2 - t1) / 1e9)}")
    if (params.timerMode) {
//      Timer.print()
    }
    postProcessed
  }
}
object ModelHolder {
  var model: InferenceModel = null
  var modelQueueing = 0
  var nonOMP = 0
}
