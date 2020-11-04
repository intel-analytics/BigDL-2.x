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

import com.intel.analytics.zoo.serving.arrow.ArrowDeserializer
import com.intel.analytics.zoo.serving.engine.{ClusterServingInference, ModelHolder}
import com.intel.analytics.zoo.serving.utils.ClusterServingHelper
import org.apache.flink.table.functions.{FunctionContext, ScalarFunction}

class ClusterServingFunction()
  extends ScalarFunction {
  val clusterServingParams = new ClusterServingParams()
  var inference: ClusterServingInference = null

  override def open(context: FunctionContext): Unit = {
    if (ModelHolder.model == null) {
      ModelHolder.synchronized {
        if (ModelHolder.model == null) {
          println("Loading Cluster Serving model...")
          val modelPath = context.getJobParameter("modelPath", "")
          require(modelPath != "", "You have not provide modelPath in job parameter.")
          val info = ClusterServingHelper
            .loadModelfromDir(modelPath, clusterServingParams._modelConcurrent)
          ModelHolder.model = info._1
          clusterServingParams._modelType = info._2
        }
      }
    }
    inference = new ClusterServingInference(null, clusterServingParams._modelType)

  }
  def eval(uri: String, data: String): String = {
    val array = data.split(" +").map(_.toFloat)
    val input = ClusterServingInput(uri, array)
    val result = inference.singleThreadInference(List((uri, input)))
    ArrowDeserializer(result.head._2)
  }
}


