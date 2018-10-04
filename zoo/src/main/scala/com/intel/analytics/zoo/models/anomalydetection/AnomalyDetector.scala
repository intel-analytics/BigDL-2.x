package com.intel.analytics.zoo.models.anomalydetection

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Shape, T}
import com.intel.analytics.zoo.models.common.ZooModel
import com.intel.analytics.zoo.pipeline.api.keras.layers.{Dense, Dropout, LSTM}
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{avg, col, udf}

import scala.reflect.ClassTag

/**
  * The anomaly detector for single time series
  *
  * @param inputShape   The input shape of features.
  * @param hiddenLayers Units of hidden layers of LSTM.
  * @param dropouts     Fraction of the input units to drop following each hidden LSTM layer. Float between 0 and 1.
  */
class AnomalyDetector[T: ClassTag](
                                    val inputShape: Shape,
                                    val hiddenLayers: Array[Int] = Array(8, 32, 15),
                                    val dropouts: Array[Float] = Array(0.2f, 0.2f, 0.2f)
                                  )
                                  (implicit ev: TensorNumeric[T])
  extends ZooModel[Tensor[T], Tensor[T], T] {

  override def buildModel(): AbstractModule[Tensor[T], Tensor[T], T] = {

    val model = Sequential[Float]()

    model.add(LSTM[Float](hiddenLayers(0), returnSequences = true, inputShape = inputShape))
      .add(Dropout[Float](dropouts(0)))

    for (i <- 1 to hiddenLayers.length - 1) {
      model.add(LSTM[Float](hiddenLayers(i), returnSequences = true))
        .add(Dropout[Float](dropouts(i)))
    }

    model.add(LSTM[Float](hiddenLayers.last, returnSequences = false))
      .add(Dropout[Float](dropouts.last))
      .add(Dense[Float](outputDim = 1))

    model.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }

}

case class FeatureLabelIndex(feature: Array[Array[Float]], label: Float, index: Long) {
  override def toString =
    "value: " + feature.map(x => x.mkString("|")).mkString(",") + " label:" + label + " index:" + index
}

object AnomalyDetector {
  /**
    * The factory method to create a NeuralCF instance.
    */
  def apply[@specialized(Float, Double) T: ClassTag](
                                                      inputShape: Shape,
                                                      hiddenLayers: Array[Int] = Array(8, 32, 15),
                                                      dropouts: Array[Float] = Array(0.2f, 0.2f, 0.2f)
                                                    )(implicit ev: TensorNumeric[T]): AnomalyDetector[T] = {
    new AnomalyDetector[T](inputShape, hiddenLayers, dropouts).build()
  }

  /**
    * Load an existing NeuralCF model (with weights).
    *
    * @param path       The path for the pre-defined model.
    *                   Local file system, HDFS and Amazon S3 are supported.
    *                   HDFS path should be like "hdfs://[host]:[port]/xxx".
    *                   Amazon S3 path should be like "s3a://bucket/xxx".
    * @param weightPath The path for pre-trained weights if any. Default is null.
    * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
    */
  def loadModel[T: ClassTag](
                              path: String,
                              weightPath: String = null)(implicit ev: TensorNumeric[T]): AnomalyDetector[T] = {
    ZooModel.loadModel(path, weightPath).asInstanceOf[AnomalyDetector[T]]
  }

  def detectAnomalies[T: ClassTag](yTruth: RDD[T], yPredict: RDD[T], anomalyFraction: Int = 5): RDD[(T, T, Any)] = {
    val totalCount = yTruth.count()

    val threshold: Float = yTruth.zip(yPredict)
      .map(x => diff(x._1, x._2))
      .sortBy(x => -x)
      .take((totalCount * anomalyFraction.toFloat / 100).toInt)
      .min

    detectAnomalies[T](yTruth, yPredict, threshold)
  }

  def detectAnomalies[T: ClassTag](yTruth: RDD[T], yPredict: RDD[T], threshold: Float): RDD[(T, T, Any)] = {
    val anomalies = yTruth.zip(yPredict).map { x =>
      val d = diff(x._1, x._2)
      val anomaly = if (d > threshold) x._1 else null
      (x._1, x._2, anomaly) //yTruth, yPredict, anomaly
    }
    anomalies
  }

  def diff[T: ClassTag](A: T, B: T): Float = {
    if (A.isInstanceOf[Float]) {
      Math.abs(A.asInstanceOf[Float] - B.asInstanceOf[Float])
    } else {
      Math.abs(A.asInstanceOf[Double] - B.asInstanceOf[Double]).toFloat
    }
  }

  def distributeUnrollAll(dataRdd: RDD[Array[Float]], unrollLength: Int, predictStep: Int = 1): RDD[FeatureLabelIndex] = {

    val n = dataRdd.count()
    val indexRdd: RDD[(Array[Float], Long)] = dataRdd.zipWithIndex()

    //RDD[index of record, feature]
    val featureRdd: RDD[(Long, Array[Array[Float]])] = indexRdd
      .flatMap(x => {
        val pairs: Seq[(Long, List[(Array[Float], Long)])] = if (x._2 < unrollLength) {
          (0L to x._2).map(index => (index, List(x)))
        } else {
          (x._2 - unrollLength + 1 to x._2).map(index => (index, List(x)))
        }
        pairs
      }).reduceByKey(_ ++ _)
      .filter(x => x._2.size == unrollLength && x._1 <= n - unrollLength - predictStep)
      .map(x => {
        val data: Array[Array[Float]] = x._2.sortBy(y => y._2).map(x => x._1).toArray
        (x._1, data)
      }).sortBy(x => x._1)

    val skipIndex: Int = unrollLength - 1 + predictStep
    val labelRdd: RDD[(Long, Float)] = indexRdd.filter(x => x._2 >= skipIndex).map(x => (x._2 - skipIndex, x._1(0)))

    val featureData: RDD[FeatureLabelIndex] = featureRdd.join(labelRdd)
      .sortBy(x => x._1)
      .map(x => FeatureLabelIndex(x._2._1, x._2._2, x._1))

    featureData.take(1).foreach(println)

    featureData
  }

  def toSampleRdd(train_data: RDD[FeatureLabelIndex]) = {

    train_data.map(x => {

      val shape: Array[Int] = Array(x.feature.length, x.feature(0).length)
      val data: Array[Float] = x.feature.flatten
      val feature: Tensor[Float] = Tensor(data, shape)
      val label = Tensor[Float](T(x.label))
      Sample(feature, label)
    })

  }

  def standardScaleHelper(df: DataFrame, colName: String) = {

    val mean = df.select(colName).agg(avg(col(colName))).collect()(0).getDouble(0)

    val stddevUdf = udf((num: Float) => (num - mean) * (num - mean))

    val stddev = Math.sqrt(df.withColumn("stddev", stddevUdf(col(colName)))
      .agg(avg(col("stddev"))).collect()(0).getDouble(0))

    println(colName + ",mean:" + mean + ",stddev:" + stddev)

    val scaledUdf = udf((num: Float) => ((num - mean) / stddev).toFloat)

    df.withColumn(colName, scaledUdf(col(colName)))
  }

  def standardScale(df: DataFrame, fields: Seq[String], index: Int = 0): DataFrame = {

    if (index == fields.length) {
      df
    } else {
      val colDf = standardScaleHelper(df, fields(index))
      standardScale(colDf, fields, index + 1)
    }
  }
}

