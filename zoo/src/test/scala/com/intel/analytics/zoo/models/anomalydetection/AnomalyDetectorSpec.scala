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

package com.intel.analytics.zoo.models.anomalydetection

import java.net.URL

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.nn.ClassNLLCriterion
import com.intel.analytics.bigdl.optim.{Adam, Optimizer, Trigger}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.MaxAbsScaler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}

import scala.util.Random

class AnomalyDetectorSpec extends ZooSpecHelper {

  var sqlContext: SQLContext = _
  var sc: SparkContext = _
  var datain: DataFrame = _
  val userCount = 100
  val itemCount = 100

  override def doBefore(): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = new SparkConf().setMaster("local[1]").setAppName("NCFTest")
    sc = NNContext.initNNContext(conf)
    sqlContext = SQLContext.getOrCreate(sc)
    val resource: URL = getClass.getClassLoader.getResource("recommender")
    datain = sqlContext.read.parquet(resource.getFile)
      .select("userId", "itemId", "label")
  }

  override def doAfter(): Unit = {
    if (sc != null) {
      sc.stop()
    }
  }

  "AnomalyDetector " should "work properly" in {
// easiest way to create dataframe is to get a case class first.
    val data = Array(
      Vectors.dense(1.0, -1, 2.0),
      Vectors.dense(2.0, 0, 0),
      Vectors.dense(0, 1, -1)
    )
    val df = sqlContext.createDataFrame(data.map(Tuple1.apply)).toDF("features")
    df.show(false)

    val scaler = new org.apache.spark.ml.feature.StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .setWithStd(true)
      .setWithMean(true)

    // Compute summary statistics by fitting the StandardScaler.
    val scalerModel = scaler.fit(df)

    // Normalize each feature to have unit standard deviation.
    val scaledData = scalerModel.transform(df)
    scaledData.show(false)

    println(scalerModel.mean.toArray.toList)
    println(scalerModel.std.toArray.toList)


    val sqlContext2 = new SQLContext(sc)
    import sqlContext2.implicits._
    val df2: DataFrame = sc.parallelize(List((1.0, -1.0, 2.0), (2.0, 0.0, 0.0), (0.0, 1.0, -1.0)))
      .toDF("c1", "c2", "c3")

    AnomalyDetector.standardScale(df2, df2.columns).show()

    val dataFrame = sqlContext.createDataFrame(Seq(
      (0, Vectors.dense(1.0, 0.1, -8.0)),
      (1, Vectors.dense(2.0, 1.0, -4.0)),
      (2, Vectors.dense(4.0, 10.0, 8.0))
    )).toDF("id", "features")

    val minMaxScaler = new MaxAbsScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")

    // Compute summary statistics and generate MaxAbsScalerModel
    val minMaxModel = minMaxScaler.fit(df)

    // rescale each feature to range [-1, 1]
    val minMaxScaled = minMaxModel.transform(df)
    minMaxScaled.select("features", "scaledFeatures").show()
  }


}