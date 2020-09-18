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
import org.apache.flink.core.execution.JobClient
import org.apache.flink.streaming.api.scala.{StreamExecutionEnvironment, _}
import org.apache.log4j.{Level, Logger}
import scopt.OptionParser


object ClusterServing {
  case class ServingParams(configPath: String = "config.yaml", testMode: Int = -1,
                           timerMode: Boolean = false)

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

  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.zoo").setLevel(Level.INFO)
  var params: SerParams = null
  val logger = Logger.getLogger(getClass)
  def run(configPath: String = "config.yaml",
          testMode: Int = -1,
          timerMode: Boolean = false): Unit = {
    val helper = new ClusterServingHelper(configPath)
    helper.initArgs()
    params = new SerParams(helper)
    params.timerMode = timerMode
    if (!helper.checkManagerYaml()) {
      println(s"ERROR - Cluster Serving with name ${helper.jobName} already exists, exited.")
      return
    }
    val serving = StreamExecutionEnvironment.getExecutionEnvironment
    serving.registerCachedFile(configPath, Conventions.SERVING_CONF_TMP_PATH)
    serving.registerCachedFile(params.modelDir, Conventions.SERVING_MODEL_TMP_DIR)
    if (testMode > 0) {
      println("Running Cluster Serving in test mode with parallelism 1")
      serving.addSource(new FlinkRedisSource(params)).setParallelism(testMode)
        .map(new FlinkInference(params)).setParallelism(testMode)
        .addSink(new FlinkRedisSink(params)).setParallelism(testMode)
    } else {
      serving.addSource(new FlinkRedisSource(params))
        .map(new FlinkInference(params))
        .addSink(new FlinkRedisSink(params))
    }

    val jobClient = serving.executeAsync("Cluster Serving - " + helper.jobName)
    val jobId = ClusterServingManager.getJobIdfromClient(jobClient)
    Runtime.getRuntime.addShutdownHook(new ShutDownThrd(helper, jobId, jobClient))
    helper.updateManagerYaml(jobId)
    while (!jobClient.getJobStatus.get().isTerminalState) {
//      println(s"Status is ${jobClient.getJobStatus.get().toString}, " +
//        s"is terminate Boolean value is ${jobClient.getJobStatus.get().isTerminalState}")
      Thread.sleep(3000)
    }


  }
  def main(args: Array[String]): Unit = {
    val param = parser.parse(args, ServingParams()).head
    run(param.configPath, param.testMode, param.timerMode)
  }
}

class ShutDownThrd(helper: ClusterServingHelper, jobId: String, jobClient: JobClient)
  extends Thread {
  override def run(): Unit = {
    println(s"Shutdown hook triggered, removing job $jobId in yaml")
    try {
      jobClient.cancel()
    } catch {
      case _: Exception =>
    }
    helper.updateManagerYaml(jobId, remove = true)
  }
}
