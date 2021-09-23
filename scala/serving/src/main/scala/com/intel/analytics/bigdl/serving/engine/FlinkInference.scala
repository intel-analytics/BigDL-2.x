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

import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.zoo.pipeline.inference.InferenceModel
import com.intel.analytics.zoo.serving.ClusterServing
import com.intel.analytics.zoo.serving.postprocessing.PostProcessing
import com.intel.analytics.zoo.serving.serialization.StreamSerializer
import com.intel.analytics.zoo.serving.utils.{ClusterServingHelper, Conventions}
import org.apache.flink.api.common.functions.RichMapFunction
import org.apache.flink.configuration.Configuration
import org.apache.log4j.Logger



class FlinkInference()
  extends RichMapFunction[List[(String, String, String)], List[(String, String)]] {

  var logger: Logger = null
  var inference: ClusterServingInference = null
  var helper: ClusterServingHelper = null

  override def open(parameters: Configuration): Unit = {
    logger = Logger.getLogger(getClass)
    helper = ClusterServing.helper
//    val t = Tensor[Float](1, 2, 3).rand()
//    val x = T.array(Array(t))
//    println(s"start directly deser --->")
//    val b = StreamSerializer.objToBytes(x)
//    val o = StreamSerializer.bytesToObj(b).asInstanceOf[Activity]
//    println(s"directly deser -> ${o.getClass}")
    try {
      if (ClusterServing.model == null) {
        ClusterServing.synchronized {
          if (ClusterServing.model == null) {
            val localModelDir = getRuntimeContext.getDistributedCache
              .getFile(Conventions.SERVING_MODEL_TMP_DIR).getPath

            logger.info(s"Model loaded at executor at path ${localModelDir}")
            helper.modelPath = localModelDir
            ClusterServing.model = ClusterServing.helper.loadInferenceModel()
          }
        }
      }
      inference = new ClusterServingInference()
    }

    catch {
      case e: Exception => logger.error(e)
       throw new Error("Model loading failed")
    }

  }

  override def map(in: List[(String, String, String)]): List[(String, String)] = {
    val t1 = System.nanoTime()
    val postProcessed = {
      if (helper.modelType == "openvino") {
        inference.multiThreadPipeline(in)
      } else {
        inference.singleThreadPipeline(in)
      }
    }

    val t2 = System.nanoTime()
    logger.info(s"${in.size} records backend time ${(t2 - t1) / 1e9} s. " +
      s"Throughput ${in.size / ((t2 - t1) / 1e9)}")
    postProcessed
  }
}

