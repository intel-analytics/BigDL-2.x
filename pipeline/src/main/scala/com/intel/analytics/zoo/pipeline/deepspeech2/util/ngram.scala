package com.intel.analytics.zoo.pipeline.deepspeech2.util

import java.io.PrintWriter

import org.apache.spark.sql.SparkSession
import scopt.OptionParser
; /**
  * Created by yuhao on 4/13/17.
  */
object ngram {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("local[4]").appName("test").getOrCreate()
    import spark.implicits._

    val n = 2
    val combines = spark.sparkContext.textFile("data/dev-clean/mapping.txt")
      .flatMap(l => l.split("\\s+").tail.sliding(n, 1).filter(_.length >= n).map(g => g.mkString("\t")))

    val grams = combines.map(g => (g, 1)).reduceByKey(_ + _).filter(_._2 > 0).collect().map(t => s"${t._2}  ${t._1}")
    val pw = new PrintWriter("model/self2.txt")

    pw.write(grams.mkString("\n"))
    pw.flush()
    pw.close()
  }

}

