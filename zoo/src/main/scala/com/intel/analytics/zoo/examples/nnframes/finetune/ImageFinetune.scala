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
package com.intel.analytics.zoo.examples.nnframes.finetune

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.CrossEntropyCriterion
import com.intel.analytics.bigdl.optim.{Top1Accuracy, Trigger}
import com.intel.analytics.bigdl.utils.{LoggerFilter, Shape}
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.pipeline.api.keras.layers._
import com.intel.analytics.zoo.pipeline.api.keras.models.Model
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.zoo.feature.image.{RowToImageFeature, _}
import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.nnframes.{NNClassifier, NNImageReader}
import org.apache.hadoop.fs.Path
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.SparkConf
import scopt.OptionParser

/**
 * Scala example for image fine tuning with Inception model on Spark DataFrame.
 * Please refer to the readme.md in the same folder for more details.
 */
object ImageFinetune {

  LoggerFilter.redirectSparkInfoLogs()

  def main(args: Array[String]): Unit = {
    Utils.trainParser.parse(args, Utils.TrainParams()).foreach(param => {
      val conf = new SparkConf().setAppName("Fine tuning Example")
      val sc = NNContext.initNNContext(conf)

      val createLabel = udf { row: Row =>
        if (new Path(row.getString(0)).getName.contains("cat")) 1.0 else 2.0
      }
      val imagesDF = NNImageReader.readImages(param.imagePath, sc,
          resizeH = 256, resizeW = 256, imageCodec = 1)
        .withColumn("label", createLabel(col("image")))
      val Array(validationDF, trainingDF) = imagesDF.randomSplit(Array(0.20, 0.80), seed = 1L)

      val featureTransformer = RowToImageFeature() ->
        ImageCenterCrop(224, 224) -> ImageChannelNormalize(123, 117, 104) ->
        ImageMatToTensor() -> ImageFeatureToTensor()
      val model = getTransferLearningModel(param.modelPath)

      val classifier = NNClassifier(model, CrossEntropyCriterion[Float](), featureTransformer)
        .setFeaturesCol("image")
        .setLearningRate(0.003)
        .setBatchSize(param.batchSize)
        .setMaxEpoch(param.nEpochs)
        .setCachingSample(false)
        .setValidation(Trigger.everyEpoch, validationDF, Array(new Top1Accuracy()), param.batchSize)

      val pipeline = new Pipeline().setStages(Array(classifier))
      val pipelineModel = pipeline.fit(trainingDF)
      val predictions = pipelineModel.transform(validationDF)

      predictions.select("image", "label", "prediction").sample(false, 0.05).show(false)
      sc.stop()
    })
  }

  private def getTransferLearningModel(preTrainedPath: String): Module[Float] = {
    // you can use Net.loadBigDL[Float](preTrainedPath).saveGraphTopology(somePath)
    // and use tensorboard to visualize the model topology and decide
    val inception = Net.loadBigDL[Float](preTrainedPath)
      .newGraph(output = "pool5/drop_7x7_s1") // remove layers after pool5/drop_7x7_s1
      .freezeUpTo("pool4/3x3_s2") // freeze layer pool4/3x3_s2 and the layers before it
      .toKeras()

    // add a new classifer
    val input = Input[Float](inputShape = Shape(3, 224, 224))
    val feature = inception.inputs(input)
    val flatten = Flatten[Float]().inputs(feature)
    val logits = Dense[Float](2).inputs(flatten)

    Model[Float](input, logits)
  }
}


private object Utils {

  case class TrainParams(
      modelPath: String = "/tmp/zoo/bigdl_inception-v1_imagenet_0.4.0.model",
      imagePath: String = "/tmp/zoo/dogs_cats/samples",
      batchSize: Int = 32,
      nEpochs: Int = 2)

  val trainParser = new OptionParser[TrainParams]("BigDL ptbModel Train Example") {
    opt[String]('m', "modelPath")
      .text("pretrained model path")
      .action((x, c) => c.copy(modelPath = x))
    opt[String]('d', "imagePath")
      .text("training data path")
      .action((x, c) => c.copy(imagePath = x))
    opt[Int]('b', "batchSize")
      .text("batchSize")
      .action((x, c) => c.copy(batchSize = x))
    opt[Int]('e', "nEpochs")
      .text("epoch numbers")
      .action((x, c) => c.copy(nEpochs = x))
  }
}
