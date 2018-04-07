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

package com.intel.analytics.zoo.pipeline.nnframes

import com.intel.analytics.bigdl.dlframes.DLModel
import com.intel.analytics.bigdl.models.inception.Inception_v1
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim.{LBFGS, Loss, Trigger}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.transform.vision.image.MatToTensor
import com.intel.analytics.bigdl.transform.vision.image.augmentation.{CenterCrop, ChannelNormalize, Resize}
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import com.intel.analytics.bigdl.visualization.{TrainSummary, ValidationSummary}
import org.apache.spark.SparkContext
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.util.Random

class NNEstimatorSpec extends FlatSpec with Matchers with BeforeAndAfter {
  var sc : SparkContext = _
  var sqlContext : SQLContext = _
  var smallData: Seq[(Array[Double], Double)] = _
  val nRecords = 100
  val maxEpoch = 20

  before {
    Random.setSeed(42)
    RNG.setSeed(42)
    val conf = Engine.createSparkConf().setAppName("Test NNEstimator").setMaster("local[1]")
    sc = SparkContext.getOrCreate(conf)
    sqlContext = new SQLContext(sc)
    smallData = NNEstimatorSpec.generateTestInput(
      nRecords, Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), -1.0, 42L)
    Engine.init
  }


  after{
    if (sc != null) {
      sc.stop()
    }
  }

  "An NNEstimator" should "has correct default params" in {
    val model = Linear[Float](10, 1)
    val criterion = ClassNLLCriterion[Float]()
    val estimator = new NNEstimator[Float](model, criterion, Array(10), Array(1))
    assert(estimator.getFeaturesCol == "features")
    assert(estimator.getLabelCol == "label")
    assert(estimator.getMaxEpoch == 50)
    assert(estimator.getBatchSize == 1)
    assert(estimator.getLearningRate == 1e-3)
    assert(estimator.getLearningRateDecay == 0)

  }

  "An NNEstimator" should "get reasonable accuracy" in {
    val model = new Sequential().add(Linear[Float](6, 2)).add(LogSoftMax[Float])
    val criterion = ClassNLLCriterion[Float]()
    val estimator = new NNEstimator[Float](model, criterion, Array(6), Array(1))
      .setBatchSize(nRecords)
      .setOptimMethod(new LBFGS[Float]())
      .setLearningRate(0.1)
      .setMaxEpoch(maxEpoch)
    val data = sc.parallelize(smallData)
    val df = sqlContext.createDataFrame(data).toDF("features", "label")

    val dlModel = estimator.fit(df)
    dlModel.isInstanceOf[DLModel[_]] should be(true)
    val correct = dlModel.transform(df).select("label", "prediction").rdd.filter {
      case Row(label: Double, prediction: Seq[_]) =>
        label == prediction.indexOf(prediction.asInstanceOf[Seq[Double]].max) + 1
    }.count()
    assert(correct > nRecords * 0.8)
  }

  "An NNEstimator" should "support different FEATURE types" in {
    val model = new Sequential().add(Linear[Float](6, 2)).add(LogSoftMax[Float])
    val criterion = ClassNLLCriterion[Float]()
    val estimator = new NNEstimator[Float](model, criterion, Array(6), Array(1))
      .setBatchSize(2)
      // intentionally set low since this only validates data format compatibility
      .setEndWhen(Trigger.maxIteration(1))

    Array(
      sqlContext.createDataFrame(sc.parallelize(smallData.map(p => (p._1, p._2))))
        .toDF("features", "label"), // Array[Double]
      sqlContext.createDataFrame(sc.parallelize(smallData.map(p => (p._1.map(_.toFloat), p._2))))
        .toDF("features", "label"), // Array[Float]
      sqlContext.createDataFrame(sc.parallelize(smallData.map(p => (Vectors.dense(p._1), p._2))))
        .toDF("features", "label") // MLlib Vector
      // TODO: add ML Vector when ut for Spark 2.0+ is ready
    ).foreach { df =>
      val dlModel = estimator.fit(df)
      dlModel.transform(df).collect()
    }
  }

  "An NNEstimator" should "support scalar FEATURE types" in {
    val model = new Sequential().add(Linear[Float](1, 2)).add(LogSoftMax[Float])
    val criterion = ClassNLLCriterion[Float]()
    val estimator = new NNEstimator[Float](model, criterion, Array(1), Array(1))
      .setBatchSize(2)
      // intentionally set low since this only validates data format compatibility
      .setEndWhen(Trigger.maxIteration(1))

    Array(
      sqlContext.createDataFrame(sc.parallelize(smallData.map(p => (p._1.head.toFloat, p._2))))
        .toDF("features", "label"), // Float
      sqlContext.createDataFrame(sc.parallelize(smallData.map(p => (p._1.head, p._2))))
        .toDF("features", "label") // Double
      // TODO: add ML Vector when ut for Spark 2.0+ is ready
    ).foreach { df =>
      val dlModel = estimator.fit(df)
      dlModel.transform(df).collect()
    }
  }

  "An NNEstimator" should "support different LABEL types" in {
    val model = new Sequential().add(Linear[Float](6, 2)).add(LogSoftMax[Float])
    val criterion = MultiLabelSoftMarginCriterion[Float]()
    val estimator = new NNEstimator[Float](model, criterion, Array(6), Array(2))
      // intentionally set low since this only validates data format compatibitliy
      .setEndWhen(Trigger.maxIteration(1))
      .setBatchSize(2)

    Array(
      sqlContext.createDataFrame(sc.parallelize(smallData.map(p => (p._1, Array(p._2, p._2)))))
        .toDF("features", "label"), // Array[Double]
      sqlContext.createDataFrame(sc.parallelize(smallData.map(p => (p._1,
        Array(p._2.toFloat, p._2.toFloat))))).toDF("features", "label"), // Array[Float]
      sqlContext.createDataFrame(sc.parallelize(smallData.map(p => (p._1,
        Vectors.dense(p._2, p._2))))).toDF("features", "label") // MLlib Vector
      // TODO: add ML Vector when ut for Spark 2.0+ is ready
    ).foreach { df =>
      val dlModel = estimator.fit(df)
      dlModel.transform(df).collect()
    }
  }

  "An NNEstimator" should "support scalar LABEL types" in {
    val model = new Sequential().add(Linear[Float](6, 2)).add(LogSoftMax[Float])
    val criterion = ClassNLLCriterion[Float]()
    val estimator = new NNEstimator[Float](model, criterion, Array(6), Array(1))
      // intentionally set low since this only validates data format compatibitliy
      .setEndWhen(Trigger.maxIteration(1))
      .setBatchSize(2)

    Array(
      sqlContext.createDataFrame(sc.parallelize(smallData.map(p => (p._1, p._2.toFloat))))
        .toDF("features", "label"), // Float
      sqlContext.createDataFrame(sc.parallelize(smallData.map(p => (p._1, p._2))))
        .toDF("features", "label") // Double
      // TODO: add ML Vector when ut for Spark 2.0+ is ready
    ).foreach { df =>
      val dlModel = estimator.fit(df)
      dlModel.transform(df).collect()
    }
  }

  "An NNEstimator" should "work with tensor data" in {

    val model = Linear[Float](10, 1)
    val criterion = ClassNLLCriterion[Float]()
    val estimator = new NNEstimator[Float](model, criterion, Array(10), Array(1))
      .setMaxEpoch(1)
      .setBatchSize(nRecords)

    val featureData = Array.tabulate(100)(_ => Tensor(10))
    val labelData = Array.tabulate(100)(_ => Tensor(1).fill(1.0f))
    val miniBatch = sc.parallelize(
      featureData.zip(labelData).map(v =>
        MinibatchData(v._1.storage.array, v._2.storage.array))
    )
    val trainingDF: DataFrame = sqlContext.createDataFrame(miniBatch).toDF("features", "label")

    val dlModel = estimator.fit(trainingDF)
    dlModel.transform(trainingDF).collect()
  }

  "An NNEstimator" should "support different batchSize" in {
    val model = new Sequential().add(Linear[Float](6, 2)).add(LogSoftMax[Float])
    val criterion = ClassNLLCriterion[Float]()
    val estimator = new NNEstimator[Float](model, criterion, Array(6), Array(1))
      .setBatchSize(51)
      .setMaxEpoch(maxEpoch)
    val data = sc.parallelize(smallData)
    val df: DataFrame = sqlContext.createDataFrame(data).toDF("features", "label")

    val dlModel = estimator.fit(df)
    dlModel.isInstanceOf[DLModel[_]] should be(true)
    dlModel.transform(df).count()
  }

  "An NNModel" should "support transform with different batchSize" in {
    val model = new Sequential().add(Linear[Float](6, 2)).add(LogSoftMax[Float])
    val criterion = ClassNLLCriterion[Float]()
    val estimator = new NNEstimator[Float](model, criterion, Array(6), Array(1))
      .setBatchSize(nRecords)
      .setMaxEpoch(maxEpoch)
    val data = sc.parallelize(smallData)
    val df: DataFrame = sqlContext.createDataFrame(data).toDF("features", "label")
    val dlModel = estimator.fit(df)
    assert(df.count() == dlModel.setBatchSize(51).transform(df).count())
  }

  "An NNEstimator" should "throws exception without correct inputs" in {
    val model = Linear[Float](10, 1)
    val criterion = ClassNLLCriterion[Float]()
    val inputs = Array[String]("Feature data", "Label data")
    var estimator = new NNEstimator[Float](model, criterion, Array(10), Array(2, 1)).
      setFeaturesCol(inputs(0)).setLabelCol(inputs(1))

    val featureData = Tensor(2, 10)
    val labelData = Tensor(2, 1)
    val miniBatch = sc.parallelize(Seq(
      MinibatchData[Float](featureData.storage().array(), labelData.storage().array())
    ))
    var df: DataFrame = sqlContext.createDataFrame(miniBatch).toDF(inputs: _*)
    // Spark 1.6 and 2.0 throws different exception here
    intercept[Exception] {
      estimator.fit(df)
    }
  }

  "An NNEstimator" should "supports training summary" in {
    val model = new Sequential().add(Linear[Float](6, 2)).add(LogSoftMax[Float])
    val criterion = ClassNLLCriterion[Float]()
    val logdir = com.google.common.io.Files.createTempDir()
    val estimator = new NNEstimator[Float](model, criterion, Array(6), Array(1))
      .setBatchSize(nRecords)
      .setMaxEpoch(5)
      .setTrainSummary(TrainSummary(logdir.getPath, "DLEstimatorTrain"))
    val data = sc.parallelize(smallData)
    val df = sqlContext.createDataFrame(data).toDF("features", "label")

    val dlModel = estimator.fit(df)
    val trainSummary = estimator.getTrainSummary.get
    val losses = trainSummary.readScalar("Loss")
    assert(losses.length == 5)
    trainSummary.close()
    logdir.deleteOnExit()
  }

  "An NNEstimator" should "supports validation data and summary" in {
    val model = new Sequential().add(Linear[Float](6, 2)).add(LogSoftMax[Float])
    val criterion = ClassNLLCriterion[Float]()
    val logdir = com.google.common.io.Files.createTempDir()
    val data = sc.parallelize(smallData)
    val df = sqlContext.createDataFrame(data).toDF("features", "label")
    val estimator = new NNEstimator[Float](model, criterion, Array(6), Array(1))
      .setBatchSize(4)
      .setEndWhen(Trigger.maxIteration(5))
      .setValidation(Trigger.severalIteration(1), df, Array(new Loss[Float]()), 2)
      .setValidationSummary(ValidationSummary(logdir.getPath, "NNEstimatorValidation"))

    val dlModel = estimator.fit(df)
    val validationSummary = estimator.getValidationSummary.get
    val losses = validationSummary.readScalar("Loss")
    assert(losses.length == 5)
    validationSummary.close()
    logdir.deleteOnExit()
  }

  "An NNEstimator" should "throws exception when EndWhen and MaxEpoch are set" in {
    val model = new Sequential().add(Linear[Float](6, 2)).add(LogSoftMax[Float])
    val criterion = ClassNLLCriterion[Float]()
    val logdir = com.google.common.io.Files.createTempDir()

    val data = sc.parallelize(smallData)
    val df = sqlContext.createDataFrame(data).toDF("features", "label")
    val estimator = new NNEstimator[Float](model, criterion, Array(6), Array(1))
      .setBatchSize(4)
      .setEndWhen(Trigger.maxIteration(5))
      .setMaxEpoch(5)

    intercept[Exception] {
      estimator.fit(df)
    }
  }

  "An NNEstimator" should "works in ML pipeline" in {
    var appSparkVersion = org.apache.spark.SPARK_VERSION
    if (appSparkVersion.trim.startsWith("1")) {
      val data = sc.parallelize(
        smallData.map(p => (org.apache.spark.mllib.linalg.Vectors.dense(p._1), p._2)))
      val df: DataFrame = sqlContext.createDataFrame(data).toDF("features", "label")

      val scaler = new MinMaxScaler().setInputCol("features").setOutputCol("scaled")
        .setMax(1).setMin(-1)
      val model = new Sequential().add(Linear[Float](6, 2)).add(LogSoftMax[Float])
      val criterion = ClassNLLCriterion[Float]()
      val estimator = new NNEstimator[Float](model, criterion, Array(6), Array(1))
        .setOptimMethod(new LBFGS[Float]())
        .setLearningRate(0.1)
        .setBatchSize(nRecords)
        .setMaxEpoch(maxEpoch)
        .setFeaturesCol("scaled")
      val pipeline = new Pipeline().setStages(Array(scaler, estimator))

      val pipelineModel = pipeline.fit(df)
      pipelineModel.isInstanceOf[PipelineModel] should be(true)
      val correct = pipelineModel.transform(df).select("label", "prediction").rdd.filter {
        case Row(label: Double, prediction: Seq[_]) =>
          label == prediction.indexOf(prediction.asInstanceOf[Seq[Double]].max) + 1
      }.count()
      assert(correct > nRecords * 0.8)
    }
  }

  "NNModel" should "support image FEATURE types" in {
    val pascalResource = getClass.getClassLoader.getResource("pascal/")
    val imageDF = NNImageReader.readImages(pascalResource.getFile, sc)
    assert(imageDF.count() == 1)
    val transformer = Resize(256, 256) -> CenterCrop(224, 224) ->
      ChannelNormalize(123, 117, 104, 1, 1, 1) -> MatToTensor()
    val transformedDF = new NNImageTransformer(transformer)
      .setInputCol("image")
      .setOutputCol("features")

    val featurizer = new NNModel[Float](
      Inception_v1(1000), Array(3, 224, 224))
      .setBatchSize(1)

    val pipeline = new Pipeline().setStages(Array(transformedDF, featurizer))
    pipeline.fit(imageDF)
  }

}

private case class MinibatchData[T](featureData : Array[T], labelData : Array[T])

object NNEstimatorSpec {
  // Generate noisy input of the form Y = signum(x.dot(weights) + intercept + noise)
  def generateTestInput(
                         numRecords: Int,
                         weight: Array[Double],
                         intercept: Double,
                         seed: Long): Seq[(Array[Double], Double)] = {
    val rnd = new Random(seed)
    val data = (1 to numRecords)
      .map( i => Array.tabulate(weight.length)(index => rnd.nextDouble() * 2 - 1))
      .map { record =>
        val y = record.zip(weight).map(t => t._1 * t._2).sum
        +intercept + 0.01 * rnd.nextGaussian()
        val label = if (y > 0) 2.0 else 1.0
        (record, label)
      }
    data
  }
}
