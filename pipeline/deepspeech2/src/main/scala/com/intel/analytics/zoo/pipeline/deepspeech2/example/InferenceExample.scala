package com.intel.analytics.zoo.pipeline.deepspeech2.example

import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.zoo.pipeline.deepspeech2.pipeline.acoustic._
import com.intel.analytics.zoo.pipeline.deepspeech2.util.{DeepSpeech2InferenceParam, parser}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.FlacReader
import org.apache.spark.sql.SparkSession

object InferenceExample {

  Logger.getLogger("org").setLevel(Level.WARN)
  val logger = Logger.getLogger(getClass)

  def main(args: Array[String]): Unit = {
    parser.parser.parse(args, DeepSpeech2InferenceParam()).foreach { param =>
      val sampleRate = 16000
      val windowSize = 400
      val windowStride = 160
      val uttLength = param.segment * (sampleRate / windowStride)
      val numFilters = 13
      logger.info(s"parameters: ${args.mkString(", ")}")

      val conf = Engine.createSparkConf()
      val spark = SparkSession.builder().config(conf).appName(this.getClass.getSimpleName).getOrCreate()
      Engine.init
      import spark.implicits._

      val audioPath = param.dataPath
      val samples = if (audioPath.toLowerCase().endsWith("flac")) {
        FlacReader.pathToSamples(audioPath)
      } else {
        WavReader.pathToSamples(audioPath)
      }
      val samplesDF = Seq((audioPath, samples)).toDF("path", "samples")

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
        .setNumFilters(numFilters)
        .setUttLength(uttLength)
      val transposeFlip = new TransposeFlip()
        .setInputCol("mel")
        .setOutputCol("features")
        .setNumFilters(numFilters)

      val modelTransformer = new DeepSpeech2ModelTransformer(param.modelPath)
        .setInputCol("features")
        .setOutputCol("prob")
        .setNumFilters(numFilters)
      val decoder = new ArgMaxDecoder()
        .setInputCol("prob")
        .setOutputCol("output")
        .setOriginalSizeCol("originalSizeCol")
        .setAlphabet("_'ABCDEFGHIJKLMNOPQRSTUVWXYZ ")
        .setUttLength(uttLength)
        .setWindowSize(windowSize)

      val pipeline = new Pipeline().setStages(
        Array(windower, dftSpecgram, melbank, transposeFlip, modelTransformer, decoder))

      val pipelineModel = pipeline.fit(samplesDF)
      val resultDF = pipelineModel.transform(samplesDF)
        .select("path", "output")

      resultDF.show(false)
    }
  }

}
