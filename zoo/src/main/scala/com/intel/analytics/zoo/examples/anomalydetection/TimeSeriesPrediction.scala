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

import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn.keras.KerasLayer
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.models.anomalydetection._
import com.intel.analytics.zoo.models.seq2seq.{RNNDecoder, RNNEncoder, Seq2seq}
import com.intel.analytics.zoo.pipeline.api.keras.layers.{Dense, InputLayer, SelectTable, TimeDistributed}
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
                           val unrollLength: Int = 50,
                           val decoderLength: Int = 10,
                           val embedDim: Int = 5,
                           val batchSize: Int = 1024,
                           val nEpochs: Int = 1
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
    val featureShape = Shape(param.unrollLength, 3)
    val unrolled: RDD[MFeatureLabelIndex[Float]] =
      assemblyFeature(featureDF, true, param.unrollLength)
    val (trainRdd, testRdd,predictRdd) = TimeSeriesPredictor.trainTestSplit(unrolled, testSize = 1000)

    val numLayers = 1
    val encoder = RNNEncoder[Float]("lstm", numLayers, param.embedDim)
    val decoder = RNNDecoder[Float]("lstm", numLayers, param.embedDim)

   val  generator = Sequential[Float]()
     .add(InputLayer[Float](inputShape = Shape(param.decoderLength, param.embedDim)))
     .add(TimeDistributed(Dense(1).asInstanceOf[KerasLayer[Activity, Tensor[Float], Float]]))

   val  autoEncoder = Seq2seq(encoder, decoder, Shape(50, 3), Shape(10, 1), null, generator)

    autoEncoder.compile(optimizer = new RMSprop(learningRate = 0.001, decayRate = 0.9),
      loss = MeanSquaredError[Float](),
      metrics = List( new MAE[Float]()))

    autoEncoder.fit(trainRdd, batchSize = param.batchSize, nbEpoch = param.nEpochs,
      validationData = testRdd)

    val features: RDD[Activity] = autoEncoder.encoder.predict(predictRdd)
    val resutlsPrint = features.take(1)

    val bridge = Sequential().add(SelectTable(1, inputShape = Shape(2)))

    resutlsPrint.map{ x =>
      println("---------------------------------------")
      println(x)
      println(x.isTable)
      println(x.isTensor)
      val out1 = bridge.forward(x)
      println(out1)

    }


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
                      unrollLength: Int): RDD[MFeatureLabelIndex[Float]] = {

    val scaledDF = if (ifScale) {
      Utils.standardScale(featureDF, Seq("value", "hour", "awake"))
    } else {
      featureDF
    }
    val featureLen = scaledDF.columns.length
    val dataRdd: RDD[Array[Float]] = scaledDF.rdd
      .map(row => (0 to featureLen - 1).toArray.map(x => row.getAs[Float](x)))

    TimeSeriesPredictor.unroll(dataRdd, 3,2)
  }

}
