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
import com.intel.analytics.zoo.serving.{ClusterServing, PreProcessing}
import com.intel.analytics.zoo.serving.postprocessing.PostProcessing
import com.intel.analytics.zoo.serving.utils.{ClusterServingHelper, Conventions}
import org.apache.flink.api.common.functions.RichMapFunction
import org.apache.flink.configuration.Configuration
import org.apache.log4j.Logger


class FlinkInference(helper: ClusterServingHelper)
  extends RichMapFunction[List[(String, String)], List[(String, String)]] {

  var logger: Logger = null
  var inference: ClusterServingInference = null

  override def open(parameters: Configuration): Unit = {
    logger = Logger.getLogger(getClass)
    try {
      if (ClusterServing.model == null) {
        ClusterServing.synchronized {
          if (ClusterServing.model == null) {
            val localModelDir = getRuntimeContext.getDistributedCache
              .getFile(Conventions.SERVING_MODEL_TMP_DIR).getPath

            logger.info(s"Model loaded at executor at path ${localModelDir}")
            helper.modelDir = localModelDir
            ClusterServing.model = helper.loadInferenceModel()
          }
        }
      }
      inference = new ClusterServingInference(new PreProcessing(
        helper.chwFlag, helper.redisHost, helper.redisPort),
        helper.modelType, helper.filter, helper.coreNum, helper.resize)
    }

    catch {
      case e: Exception => logger.error(e)
       throw new Error("Model loading failed")
    }

  }

  override def map(in: List[(String, String)]): List[(String, String)] = {
    val t1 = System.nanoTime()
    val postProcessed = {
      if (helper.modelType == "openvino") {
        inference.multiThreadPipeline(in)
      } else {
        inference.singleThreadPipeline(in)
      }
    }

    val t2 = System.nanoTime()
    logger.debug(s"${in.size} records backend time ${(t2 - t1) / 1e9} s. " +
      s"Throughput ${in.size / ((t2 - t1) / 1e9)}")
    postProcessed
  }
}

