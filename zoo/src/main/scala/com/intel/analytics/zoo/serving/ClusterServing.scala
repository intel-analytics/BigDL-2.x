package com.intel.analytics.zoo.serving

import com.intel.analytics.zoo.serving.engine.{FlinkInference, FlinkRedisSink, FlinkRedisSource}
import com.intel.analytics.zoo.serving.utils.{ClusterServingHelper, SerParams}
import org.apache.flink.api.scala.DataSet
import org.apache.flink.streaming.api.scala.{StreamExecutionEnvironment, _}
import org.apache.log4j.{Level, Logger}

import scala.collection.JavaConverters._

object ClusterServing {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.zoo").setLevel(Level.INFO)
  var params: SerParams = null

  def main(args: Array[String]): Unit = {
    val helper = new ClusterServingHelper()
    helper.initArgs()
    params = new SerParams(helper)

    val serving = StreamExecutionEnvironment.getExecutionEnvironment
    serving.addSource(new FlinkRedisSource())
      .map(new FlinkInference())
      .addSink(new FlinkRedisSink())
    serving.execute("Cluster Serving - Flink")
  }
}
