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

package com.intel.analytics.zoo.pipeline.example.nnframes.ImageTransferLearning

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.transform.vision.image._
import com.intel.analytics.bigdl.transform.vision.image.augmentation._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.{Engine, LoggerFilter}
import com.intel.analytics.zoo.pipeline.nnframes._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.SparkContext
import scopt.OptionParser

object ImageTransferLearning {
  LoggerFilter.redirectSparkInfoLogs()

  def main(args: Array[String]): Unit = {

    val defaultParams = Utils.LocalParams()

    Utils.parser.parse(args, defaultParams).map { params =>

      val conf = Engine.createSparkConf().setAppName("TransferLearning")
      val sc = SparkContext.getOrCreate(conf)
      val sqlContext = new SQLContext(sc)
      Engine.init

      val createLabel = udf { row: Row => if (row.getString(0).contains("cat")) 1.0 else 2.0 }
      val imagesDF: DataFrame = NNImageReader.readImages(params.folder, sqlContext.sparkContext)
        .withColumn("label", createLabel(col("image")))

      val Array(validationDF, trainingDF) = imagesDF.randomSplit(Array(0.20, 0.80), seed = 1L)

      validationDF.persist()
      trainingDF.persist()

      val featureTransformer = new NNImageTransformer(
        Resize(256, 256) ->
        CenterCrop(224, 224) ->
        ChannelNormalize(123, 117, 104) ->
        MatToTensor() ->
        ImageFrameToSample())

      val loadedModel = Module
        .loadCaffeModel[Float](params.caffeDefPath, params.modelPath)

      val featurizer = new
          NNModel[Float](loadedModel, Array(3, 224, 224))
        .setBatchSize(params.batchSize)
        .setFeaturesCol("output")
        .setPredictionCol("embedding")

      val lrModel = Sequential().add(Linear(1000, 2)).add(LogSoftMax())

      val classifier = new NNClassifier(lrModel, ClassNLLCriterion[Float](), Array(1000))
        .setFeaturesCol("embedding")
        .setLearningRate(0.003)
        .setBatchSize(params.batchSize)
        .setMaxEpoch(params.nEpochs)

      val pipeline = new Pipeline().setStages(Array(featureTransformer, featurizer, classifier))

      val pipelineModel = pipeline.fit(trainingDF)
      trainingDF.unpersist()

      val predictions = pipelineModel.transform(validationDF)

      predictions.show(20)

      val evaluation = new MulticlassClassificationEvaluator().setPredictionCol("prediction")
        .setMetricName("weightedPrecision").evaluate(predictions)
      println("evaluation result on validationDF: " + evaluation)

      validationDF.unpersist()
    }
  }

}


object Utils {

  case class LocalParams(
    caffeDefPath: String = " ",
    modelPath: String = " ",
    folder: String = " ",
    batchSize: Int = 16,
    nEpochs: Int = 10)

  val defaultParams = LocalParams()

  val parser = new OptionParser[LocalParams]("Analytics zoo image transfer learning Example") {
    opt[String]("caffeDefPath")
      .text(s"caffeDefPath")
      .action((x, c) => c.copy(caffeDefPath = x))
    opt[String]("modelPath")
      .text(s"modelPath")
      .action((x, c) => c.copy(modelPath = x))
    opt[String]("folder")
      .text(s"folder")
      .action((x, c) => c.copy(folder = x))
    opt[Int]('b', "batchSize")
      .text(s"batchSize")
      .action((x, c) => c.copy(batchSize = x.toInt))
    opt[Int]('e', "nEpochs")
      .text("epoch numbers")
      .action((x, c) => c.copy(nEpochs = x))
  }
}
