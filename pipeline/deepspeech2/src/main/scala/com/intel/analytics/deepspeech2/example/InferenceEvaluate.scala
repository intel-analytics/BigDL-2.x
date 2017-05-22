package com.intel.analytics.deepspeech2.example

import java.nio.file.Paths

import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.deepspeech2.pipeline.acoustic._
import com.intel.analytics.deepspeech2.util.{LocalOptimizerPerfParam, parser}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.FlacReader
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

/**
 * load trained model to inference the audio files.
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

      val conf = Engine.createSparkConf().set("spark.driver.maxResultSize", "20g")
      val spark = SparkSession.builder().config(conf).appName(this.getClass.getSimpleName).getOrCreate()
      Engine.init

      val df = dataLoader(spark, param.dataPath, param.numFile, param.partition)
      df.repartition(param.partition)
      logger.info(s"${df.count()} audio files, in ${df.rdd.partitions.length} partitions")
      df.show()

      val st = System.nanoTime()
      val pipeline = getPipeline(param.modelPath, uttLength, windowSize, windowStride, numFilters, sampleRate, param.segment)
      val model = pipeline.fit(df)
      val result = model.transform(df).select("path", "output", "target").cache()
      logger.info("inference finished: " + (System.nanoTime() - st) / 1e9)
      logger.info(s"evaluation result:")
      result.select("output", "target").rdd.collect().foreach { case Row(output, target) =>
        logger.info(s"output: $output")
        logger.info(s"target: $target")
      }

      evaluate(model, df, param.segment)
      logger.info("total time = " + (System.nanoTime() - st) / 1e9)
    }
  }

  private def dataLoader(spark: SparkSession, path: String, takeNum: Int, partitionNum: Int): DataFrame = {
    val sc = spark.sparkContext

    import spark.implicits._
    if (path.toLowerCase().startsWith("hdfs")) {
      logger.info("load data from hdfs ..")
      val paths = sc.textFile(path + "/mapping.txt")
        .take(takeNum)
        .map { line =>
          val firstSpace = line.indexOf(" ")
          val audioPath = path + "/" + line.substring(0, firstSpace) + ".flac"
          val transcript = line.substring(line.indexOf(" ") + 1)
          (audioPath, transcript)
        }
      val pathsDF = spark.createDataset(paths).toDF("path", "target").cache()
      val flacReader = new FlacReader().setInputCol("path").setOutputCol("samples")
      val samplesDF = flacReader.transform(pathsDF)
      samplesDF.repartition(partitionNum)
    } else {
      // assume files are stored locally
      logger.info("load data from local disk ..")
      val paths = sc.textFile(Paths.get(path, "/mapping.txt").toString)
        .take(takeNum)
        .map { line =>
          val firstSpace = line.indexOf(" ")
          val audioPath = Paths.get(path, line.substring(0, firstSpace)).toString + ".flac"
          val transcript = line.substring(line.indexOf(" ") + 1)
          (audioPath, transcript)
        }

      val seq = paths.map { case (audioPath, transcirpt) =>
        val samples = if (audioPath.toLowerCase().endsWith("flac")) {
          FlacReader.pathToSamples(audioPath)
        } else {
          WavReader.pathToSamples(audioPath)
        }
        (audioPath, samples, transcirpt)
      }
      spark.createDataset(seq).toDF("path", "samples", "target")
        .repartition(partitionNum)
    }
  }

  private def getPipeline(modelPath: String, uttLength: Int, windowSize: Int,
      windowStride: Int, numFilter: Int, sampleRate: Int, isSegment: Boolean): Pipeline = {

    val segmenter = new TimeSegmenter()
      .setSegmentSize(sampleRate * 60)
      .setInputCol("samples")
      .setOutputCol("segments")
    val windower = new Windower()
      .setInputCol("segments")
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

    val pipeline = new Pipeline()

    if (isSegment) {
      pipeline.setStages(
        Array(segmenter, windower, dftSpecgram, melbank, transposeFlip, modelTransformer, decoder))
    } else {
      windower
        .setInputCol("samples")
      pipeline.setStages(
        Array(windower, dftSpecgram, melbank, transposeFlip, modelTransformer, decoder))
    }

    pipeline
  }

  private def evaluate(model: PipelineModel, df: DataFrame, isSegment: Boolean): Unit = {

    val result = if (isSegment) {
      val results = model.transform(df).select("path", "target", "audio_id", "audio_seq", "output").cache()
      results.select("path", "audio_id", "audio_seq", "output").show(false)

      val grouped = results.rdd.map {
        case Row(path: String, target: String, audio_id: Long, audio_seq: Int, output: String) =>
        (audio_id, (path, target, audio_seq, output))
      }.groupByKey()
        .map(_._2)
        .map { iter =>
          val path = iter.head._1
          val target = iter.head._2
          val text = iter.toArray.sortBy(_._3).map(_._4).mkString(" ")
          (path, text, target)
        }

      val spark = df.sparkSession
      import spark.implicits._
      spark.createDataset(grouped).toDF("path", "output", "target").cache()
    } else {
      model.transform(df).select("path", "output", "target").cache()
    }

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
