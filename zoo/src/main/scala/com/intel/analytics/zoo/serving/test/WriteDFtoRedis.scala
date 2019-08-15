package com.intel.analytics.zoo.test

import org.apache.spark.sql.SparkSession

object WriteDFtoRedis {
  case class Person(name: String, age: String)

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName("redis-df")
      .master("local[*]")
      .config("spark.redis.host", "localhost")
      .config("spark.redis.port", "6379")
      .getOrCreate()

    val personSeq = Seq(Person("img", "1.0|0.0|"))
    val df = spark.createDataFrame(personSeq)

    df.write
      .format("org.apache.spark.sql.redis")
      .option("table", "test")
      .save()
  }
}
