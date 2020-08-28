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


package com.intel.analytics.zoo.serving


import com.intel.analytics.zoo.serving.engine.{FlinkInference, FlinkRedisSink, FlinkRedisSource}
import com.intel.analytics.zoo.serving.utils.{ClusterServingHelper, ClusterServingManager, Conventions, SerParams}
import org.apache.flink.streaming.api.scala.{StreamExecutionEnvironment, _}
import org.apache.log4j.{Level, Logger}
import scopt.OptionParser


object ClusterServing {
  case class ServingParams(configPath: String = "config.yaml", testMode: Boolean = false,
                           sourceNum: Int = 1)

  val parser = new OptionParser[ServingParams]("Text Classification Example") {
    opt[String]('c', "configPath")
      .text("Config Path of Cluster Serving")
      .action((x, params) => params.copy(configPath = x))
    opt[Boolean]('t', "testMode")
      .text("Text Mode of Parallelism 1")
      .action((x, params) => params.copy(testMode = x))
  }

  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.zoo").setLevel(Level.INFO)
  var params: SerParams = null
  val logger = Logger.getLogger(getClass)
  def run(configPath: String = "config.yaml", testMode: Boolean = false): Unit = {
    val helper = new ClusterServingHelper(configPath)
    helper.initArgs()
    params = new SerParams(helper)
    val serving = StreamExecutionEnvironment.getExecutionEnvironment
    serving.registerCachedFile(configPath, Conventions.SERVING_CONF_TMP_PATH)
    serving.registerCachedFile(params.modelDir, Conventions.SERVING_MODEL_TMP_DIR)
    if (testMode) {
      println("Running Cluster Serving in test mode with parallelism 1")
      serving.addSource(new FlinkRedisSource(params)).setParallelism(1)
        .map(new FlinkInference(params)).setParallelism(1)
        .addSink(new FlinkRedisSink(params)).setParallelism(1)
    } else {
      serving.addSource(new FlinkRedisSource(params))
        .map(new FlinkInference(params))
        .addSink(new FlinkRedisSink(params))
    }

    val jobClient = serving.executeAsync()
    ClusterServingManager.writeObjectToFile(jobClient)
//    serving.execute("Cluster Serving - Flink")
  }
  def main(args: Array[String]): Unit = {
    val param = parser.parse(args, ServingParams()).head
    run(param.configPath, param.testMode)
  }
}
