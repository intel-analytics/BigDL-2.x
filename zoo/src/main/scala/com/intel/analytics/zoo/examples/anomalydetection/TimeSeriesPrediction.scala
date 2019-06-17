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

package com.intel.analytics.zoo.examples.anomalydetection

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.nn.SelectTable
import com.intel.analytics.bigdl.nn.{Sequential => TSequential}
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn.keras.KerasLayer
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.models.anomalydetection._
import com.intel.analytics.zoo.models.seq2seq.{RNNDecoder, RNNEncoder, Seq2seq}
import com.intel.analytics.zoo.pipeline.api.keras.layers.{Dense, InputLayer, TimeDistributed}
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.objectives.MeanSquaredError
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.sql.functions._
import scopt.OptionParser
import org.joda.time.format.DateTimeFormat


case class PredictorParams(val inputDir: String = "./data/NAB/nyc_taxi/",
                           val encoderLength: Int = 50,
                           val decoderLength: Int = 10,
                           val hiddenSize: Int = 80,
                           val batchSize: Int = 1024,
                           val nEpochs: Int = 20,
                           val testSize: Int = 1000
                          )

object TimeSeriesPrediction {

  def main(args: Array[String]): Unit = {

    val defaultParams = PredictorParams()

    val parser = new OptionParser[PredictorParams]("NCF Example") {
      opt[String]("inputDir")
        .text(s"inputDir")
        .action((x, c) => c.copy(inputDir = x))
      opt[Int]('b', "batchSize")
        .text(s"batchSize")
        .action((x, c) => c.copy(batchSize = x.toInt))
      opt[Int]('e', "nEpochs")
        .text("epoch numbers")
        .action((x, c) => c.copy(nEpochs = x))
    }

    parser.parse(args, defaultParams).map {
      params =>
        run(params)
    } getOrElse {
      System.exit(1)
    }
  }

  def run(param: PredictorParams): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = new SparkConf()
    conf.setAppName("AnomalyDetectionExample")
    val sc = NNContext.initNNContext(conf)
    val sqlContext = SQLContext.getOrCreate(sc)
    val featureDF = loadData(sqlContext, param.inputDir)
    val featureShape = Shape(param.encoderLength, 3)
    val unrolled: RDD[MFeatureLabelIndex[Float]] =
      assemblyFeature(featureDF, true, param.encoderLength, param.decoderLength)
    val (trainRdd, testRdd, encoderRdd) = TimeSeriesPredictor.trainTestSplit(unrolled, param.testSize)

    val numLayers = 2
    val encoder = RNNEncoder[Float]("lstm", numLayers, param.hiddenSize)
    val decoder = RNNDecoder[Float]("lstm", numLayers, param.hiddenSize)

   val  generator = Sequential[Float]()
     .add(InputLayer[Float](inputShape = Shape(param.decoderLength, param.hiddenSize)))
     .add(TimeDistributed(Dense(1).asInstanceOf[KerasLayer[Activity, Tensor[Float], Float]]))

   val  autoEncoder = Seq2seq(encoder, decoder, Shape(param.encoderLength, 3),
     Shape(param.decoderLength, 1), null, generator)

    autoEncoder.compile(optimizer = new RMSprop(learningRate = 0.001, decayRate = 0.9),
      loss = MeanSquaredError[Float](),
      metrics = List( new MAE[Float]()))

    autoEncoder.fit(trainRdd, batchSize = param.batchSize, nbEpoch = param.nEpochs,
      validationData = testRdd)

    val features: RDD[Activity] = autoEncoder.encoder.predict(encoderRdd)
    val resutlsPrint = features.take(1)

    val module = TSequential().add(SelectTable(2)).add(SelectTable(1)).add(SelectTable(2))

    val encoderStates = features.map{ x => module.forward(x) }
    val labels = encoderRdd.map { x => x.label.valueAt(1) }

    val samplesForPredictor = encoderStates.zip(labels).map{ x=>
      val feature = x._1.toTensor
      val sample = Sample(feature, x._2)
      print(sample)
      sample
    }.zipWithIndex()

    val predictor = TimeSeriesPredictor[Float](Shape(param.hiddenSize))
    predictor.compile(optimizer = new Adam[Float](learningRate = 0.001, learningRateDecay = 1e-6),
      loss = MeanSquaredError[Float](),
      metrics = List( new MAE[Float]()))

    val cutPoint = unrolled.count() - param.testSize

    val train2 = samplesForPredictor.filter(x=> x._2 <= cutPoint ).map( x=> x._1)
    val test2 = samplesForPredictor.filter(x=> x._2 > cutPoint ).map( x=> x._1)

    predictor.fit(train2,batchSize = param.batchSize, nbEpoch = param.nEpochs,
      validationData = test2 )

  }


  def loadData(sqlContext: SQLContext, dataPath: String): DataFrame = {

    @transient lazy val formatter = DateTimeFormat.forPattern("yyyy-MM-dd HH:mm:ss")
    import sqlContext.implicits._

    val df = sqlContext.sparkContext.textFile(dataPath + "/nyc_taxi.csv")
      .mapPartitionsWithIndex((idx, iter) => if (idx == 0) iter.drop(1) else iter)
      .map(x => {
        val line = x.split(",")
        Taxi(line(0), line(1).toFloat)
      }).toDF()

    val hourUDF = udf((time: String) => (formatter.parseDateTime(time).hourOfDay().get()))
    val awakeUDF = udf((hour: Int) => if (hour >= 6 && hour <= 23) 1 else 0)
    val featureDF = df.withColumn("hour", hourUDF(col("ts")))
      .withColumn("awake", awakeUDF(col("hour")))
      .select("value", "hour", "awake")

    featureDF
  }

  def assemblyFeature(featureDF: DataFrame,
                      ifScale: Boolean = true,
                      encoderLength: Int,
                      decoderLength:Int): RDD[MFeatureLabelIndex[Float]] = {

    val scaledDF = if (ifScale) {
      Utils.standardScale(featureDF, Seq("value", "hour", "awake"))
    } else {
      featureDF
    }

    val featureLen = scaledDF.columns.length
    val dataRdd: RDD[Array[Float]] = scaledDF.rdd
      .map(row => (0 to featureLen - 1).toArray.map(x => row.getAs[Float](x)))

    TimeSeriesPredictor.unroll(dataRdd, encoderLength, decoderLength)
  }

}
