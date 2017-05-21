package com.intel.analytics.zoo.pipeline.deepspeech2.example

import java.nio.file.Paths

import com.intel.analytics.zoo.pipeline.deepspeech2.pipeline.acoustic._
import com.intel.analytics.util.{LocalOptimizerPerfParam, parser}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.FlacReader
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
 * load trained model to inference the audio files.
 * sample arguments: --modelPath model --dataPath dataSample/dev-clean -n 4 -b 4
 */
object InferenceEvaluate {

  Logger.getLogger("org").setLevel(Level.WARN)
  val logger = Logger.getLogger(getClass)

  def main(args: Array[String]): Unit = {

    parser.parser.parse(args, LocalOptimizerPerfParam()).foreach { param =>
      val sampleRate = 16000
      val windowSize = 400
      val windowStride = 160
      val uttLength = 30 * (sampleRate / windowStride)
      val numFilters = 13
      logger.info(s"parameters: ${args.mkString(", ")}")

      val spark = SparkSession.builder().appName(this.getClass.getSimpleName).getOrCreate()
      val st = System.nanoTime()

      val df = textLoad(spark, param.dataPath, param.numFile)
      logger.info(s"${df.count()} audio files, in ${df.rdd.partitions.length} partitions")
      df.show()

      val pipeline = getPipeline(param.modelPath, uttLength, windowSize, windowStride, numFilters)
      val model = pipeline.fit(df)
      evaluate(model, df)

      logger.info("total time = " + (System.nanoTime() - st) / 1e9)
    }
  }

  private def textLoad(spark: SparkSession, path: String, takeNum: Int): DataFrame = {
    val sc = spark.sparkContext
    val paths = sc.textFile(Paths.get(path, "/mapping.txt").toString)
      .take(takeNum)
      .map { line =>
        val firstSpace = line.indexOf(" ")
        val audioPath = Paths.get(path, line.substring(0, firstSpace)).toString
        val transcript = line.substring(line.indexOf(" ") + 1)
        (audioPath, transcript)
      }

    import spark.implicits._
    if (path.toLowerCase().startsWith("hdfs")) {
      val pathsDF = spark.createDataset(paths).toDF("path", "target").cache()
      val flacReader = new FlacReader().setInputCol("path").setOutputCol("samples")
      val samplesDF = flacReader.transform(pathsDF)
      samplesDF
    } else { // assume files are stored locally
      val seq = paths.map { case (audioPath, transcirpt) =>
        val samples = if (audioPath.toLowerCase().endsWith("flac")) {
          FlacReader.pathToSamples(audioPath)
        } else {
          WavReader.pathToSamples(audioPath)
        }
        (audioPath, samples, transcirpt)
      }
      spark.createDataset(seq).toDF("path", "samples", "target")
    }
  }

  private def getPipeline(modelPath: String, uttLength: Int, windowSize: Int,
      windowStride: Int, numFilter: Int): Pipeline = {

    val windower = new Windower()
      .setInputCol("samples")
      .setOutputCol("window")
      .setOriginalSizeCol("originalSizeCol")
      .setWindowShift(windowStride)
      .setWindowSize(windowSize)
    val dftSpecgram = new DFTSpecgram()
      .setInputCol("window")
      .setOutputCol("specgram")
      .setWindowSize(windowSize)
    val melbank = new MelFrequencyFilterBank()
      .setInputCol("specgram")
      .setOutputCol("mel")
      .setWindowSize(windowSize)
      .setNumFilters(numFilter)
      .setUttLength(uttLength)
    val transposeFlip = new TransposeFlip()
      .setInputCol("mel")
      .setOutputCol("features")
      .setNumFilters(numFilter)

    val modelTransformer = new DeepSpeech2ModelTransformer(modelPath)
      .setInputCol("features")
      .setOutputCol("prob")
      .setNumFilters(numFilter)
    val decoder = new ArgMaxDecoder()
      .setInputCol("prob")
      .setOutputCol("output")
      .setOriginalSizeCol("originalSizeCol")
      .setAlphabet("_'ABCDEFGHIJKLMNOPQRSTUVWXYZ ")
      .setUttLength(uttLength)
      .setWindowSize(windowSize)

    new Pipeline().setStages(
      Array(windower, dftSpecgram, melbank, transposeFlip, modelTransformer, decoder))
  }

  private def evaluate(model: PipelineModel, df: DataFrame): Unit = {
    val result = model.transform(df).select("path", "output", "target").cache()
    logger.info(s"evaluation result:")
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

}
