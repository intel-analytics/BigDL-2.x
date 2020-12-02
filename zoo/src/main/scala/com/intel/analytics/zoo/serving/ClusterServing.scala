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


import com.intel.analytics.zoo.pipeline.inference.InferenceModel
import com.intel.analytics.zoo.serving.engine.{FlinkInference, FlinkRedisSink, FlinkRedisSource}
import com.intel.analytics.zoo.serving.utils.{ClusterServingHelper, Conventions}
import org.apache.flink.core.execution.JobClient
import org.apache.flink.streaming.api.scala.{StreamExecutionEnvironment, _}
import org.apache.log4j.{Level, Logger}
import redis.clients.jedis.JedisPool
import scopt.OptionParser


object ClusterServing {
  case class ServingParams(configPath: String = "config.yaml", testMode: Int = -1,
                           timerMode: Boolean = false)
  val logger = Logger.getLogger(getClass)
  var argv: ServingParams = _
  var helper: ClusterServingHelper = _
  var streamingEnv: StreamExecutionEnvironment = _
  var model: InferenceModel = _
  var jedisPool: JedisPool = _
  val parser = new OptionParser[ServingParams]("Text Classification Example") {
    opt[String]('c', "configPath")
      .text("Config Path of Cluster Serving")
      .action((x, params) => params.copy(configPath = x))
    opt[Int]('t', "testMode")
      .text("Text Mode of Parallelism 1")
      .action((x, params) => params.copy(testMode = x))
    opt[Boolean]("timerMode")
      .text("Whether to open timer mode")
      .action((x, params) => params.copy(timerMode = x))
  }
  def uploadModel(): Unit = {
    streamingEnv = StreamExecutionEnvironment.getExecutionEnvironment
    streamingEnv.registerCachedFile(helper.modelDir, Conventions.SERVING_MODEL_TMP_DIR)
  }
  def executeJob(): Unit = {
    if (argv.testMode > 0) {
      Logger.getLogger("org").setLevel(Level.INFO)
      Logger.getLogger("com.intel.analytics.zoo").setLevel(Level.DEBUG)
      streamingEnv.addSource(new FlinkRedisSource(helper)).setParallelism(argv.testMode)
        .map(new FlinkInference(helper)).setParallelism(argv.testMode)
        .addSink(new FlinkRedisSink(helper)).setParallelism(argv.testMode)
    } else {
      streamingEnv.addSource(new FlinkRedisSource(helper))
        .map(new FlinkInference(helper))
        .addSink(new FlinkRedisSink(helper))
    }
    streamingEnv.executeAsync()
  }

  def main(args: Array[String]): Unit = {
    argv = parser.parse(args, ServingParams()).head
    helper = new ClusterServingHelper()
    helper.loadConfig()
    uploadModel()
    executeJob()
  }
}
