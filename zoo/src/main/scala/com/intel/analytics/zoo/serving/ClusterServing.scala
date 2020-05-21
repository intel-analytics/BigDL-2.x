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
import com.intel.analytics.zoo.serving.utils.{ClusterServingHelper, SerParams}
import org.apache.flink.streaming.api.scala.{StreamExecutionEnvironment, _}
import org.apache.log4j.{Level, Logger}

import scala.collection.JavaConverters._


object ClusterServing {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.zoo").setLevel(Level.INFO)
  var params: SerParams = null
  def run(configPath: String = "config.yaml",
          redisHost: String = null, redisPort: Int = -1): Unit = {
    val helper = new ClusterServingHelper(configPath)
    helper.initArgs()
    params = new SerParams(helper)
    if (redisHost != null) {
      params.redisHost = redisHost
    }
    if (redisPort != -1) {
      params.redisPort = redisPort
    }
//    println(params.model)
    val serving = StreamExecutionEnvironment.getExecutionEnvironment
    serving.addSource(new FlinkRedisSource(params))
      .map(new FlinkInference(params))
      .addSink(new FlinkRedisSink(params)).setParallelism(1)
    serving.setParallelism(1)
    serving.execute("Cluster Serving - Flink")
  }
  def main(args: Array[String]): Unit = {
    run()
//    run(redisHost = "10.239.47.210", redisPort = 16380)
  }
}
