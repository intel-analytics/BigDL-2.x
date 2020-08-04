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


import java.io.File

import com.intel.analytics.zoo.serving.engine.{FlinkInference, FlinkRedisSink, FlinkRedisSource}
import com.intel.analytics.zoo.serving.utils.{ClusterServingHelper, ClusterServingManager, FileUtils, SerParams}
import org.apache.flink.streaming.api.scala.{StreamExecutionEnvironment, _}
import org.apache.log4j.{Level, Logger}
import scopt.OptionParser

import scala.util.control.Breaks._
import scala.collection.JavaConverters._



object ClusterServing {
  case class ServingParams(configPath: String = "config.yaml")

  val parser = new OptionParser[ServingParams]("Text Classification Example") {
    opt[String]('c', "configPath")
      .text("Config Path of Cluster Serving")
      .action((x, params) => params.copy(configPath = x))
  }

  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.zoo").setLevel(Level.INFO)
  var params: SerParams = null
  val logger = Logger.getLogger(getClass)
  def run(configPath: String = "config.yaml"): Unit = {
    val helper = new ClusterServingHelper(configPath)
    helper.initArgs()
    params = new SerParams(helper)
    val serving = StreamExecutionEnvironment.getExecutionEnvironment
    serving.addSource(new FlinkRedisSource(params))
      .map(new FlinkInference(params))
      .addSink(new FlinkRedisSink(params))
    val jobClient = serving.executeAsync()
    ClusterServingManager.writeObjectToFile(jobClient)
//    serving.execute("Cluster Serving - Flink")
  }
  def main(args: Array[String]): Unit = {
    val param = parser.parse(args, ServingParams()).head
    run(param.configPath)
  }
}
