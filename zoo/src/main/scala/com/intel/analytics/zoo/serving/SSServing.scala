package com.intel.analytics.zoo.serving

import org.apache.spark.sql.SparkSession
import org.apache.spark.streaming.{Duration, StreamingContext}
import com.redislabs.provider.redis.streaming._
import org.apache.log4j.{Level, Logger}

object SSServing {
  Logger.getLogger("org").setLevel(Level.ERROR)
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .master("local[2]")
      .config("spark.redis.host", "localhost")
      .config("spark.redis.port", "6379")
      .getOrCreate()

    val ssc = new StreamingContext(spark.sparkContext, new Duration(1000))

    val image = ssc.createRedisXStream(Seq(ConsumerConfig("image_stream", "group", "cli")))
    image.foreachRDD(x => {
//      println(s"time 1 ${System.currentTimeMillis()}")
      val c = x.count()
      println(c.toString)
//      println(s"time 2 ${System.currentTimeMillis()}")
    })
    ssc.start()
    ssc.awaitTermination()
  }

}
