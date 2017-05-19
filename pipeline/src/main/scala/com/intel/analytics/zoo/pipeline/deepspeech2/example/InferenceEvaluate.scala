package com.intel.analytics.zoo.pipeline.deepspeech2.example

import java.nio.file.Paths

import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.zoo.pipeline.deepspeech2.pipeline.acoustic._
import com.intel.analytics.util.{LocalOptimizerPerfParam, parser}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.FlacReader
import org.apache.spark.sql.{DataFrame, SparkSession}

object InferenceEvaluate {

  Logger.getLogger("org").setLevel(Level.WARN)
  val logger = Logger.getLogger(getClass)

  def textLoad(sc: SparkContext, host: String, path: String, takeNum: Int)
  : Array[(String, String)] = {

    sc.textFile(host + Paths.get(path, "/dev-clean/mapping.txt").toString)
      .filter(_.startsWith("1462-170142")).take(takeNum)
      .map { line =>
        val firstSpace = line.indexOf(" ")
        val audioPath = host + Paths.get(path, "/dev-clean/" +
          line.substring(0, firstSpace) + ".flac").toString
        val transcript = line.substring(line.indexOf(" ") + 1)
        (audioPath, transcript)
      }
  }

  def pipelineBuild(host: String, path: String): Pipeline = {

    val flacReader = new FlacReader(host).setInputCol("path").setOutputCol("samples")
    val windower = new Windower().setInputCol("samples").setOutputCol("window")
      .setWindowShift(160).setWindowSize(400)
    val dFTSpecgram = new DFTSpecgram().setInputCol("window")
      .setOutputCol("specgram").setWindowSize(400)
    val melbank = new MelFrequencyFilterBank().setInputCol("specgram")
      .setOutputCol("mel").setWindowSize(400).setNumFilters(13).setUttLength(3000)
    val transposeFlip = new TransposeFlip().setInputCol("mel")
      .setOutputCol("features").setNumFilters(13)

    val modelTransformer = new DeepSpeech2ModelTransformer(host + path)
      .setInputCol("features").setOutputCol("prob").setNumFilters(13)
    val decoder = new ArgMaxDecoder().setInputCol("prob")
      .setOutputCol("output").setAlphabet("_'ABCDEFGHIJKLMNOPQRSTUVWXYZ ")
      .setUttLength(3000).setWindowSize(400)

    val pipeline = new Pipeline()
    pipeline.setStages(Array(flacReader, windower, dFTSpecgram, melbank, transposeFlip
      , modelTransformer, decoder
    ))
  }

  def evaluate(model: PipelineModel, df: DataFrame): Unit = {
    val result = model.transform(df).select("path", "output", "target").cache()
    result.select("output", "target").rdd.map { r =>
      val output = r.getString(0)
      val target = r.getString(1)
      (output, target)
    }.collect().foreach { case (output, target) =>
      logger.info(s"output: $output")
      logger.info(s"target: $target")
    }

    val cer = new ASREvaluator().setLabelCol("target")
      .setPredictionCol("output").evaluate(result)
    logger.info("cer = " + cer)
    val wer = new ASREvaluator().setLabelCol("target")
      .setPredictionCol("output").setMetricName("wer").evaluate(result)
    logger.info("wer = " + wer)
  }

  def main(args: Array[String]): Unit = {

    parser.parser.parse(args, new LocalOptimizerPerfParam()).map(param => {

      val conf = Engine.createSparkConf()
      val spark = SparkSession.builder().appName("test").getOrCreate()
      import spark.implicits._

      val st = System.nanoTime()

      val samples = textLoad(spark.sparkContext, param.host, param.path, takeNum = 2)

      val df = spark.createDataset(samples).repartition(8).toDF("path", "target").cache()

      logger.info(df.count() + " audio files")
      df.show()

      val pipeline = pipelineBuild(param.host, param.path)

      val model = pipeline.fit(df)

      evaluate(model, df)

      logger.info("total time = " + (System.nanoTime() - st) / 1e9)
    })
  }
}
