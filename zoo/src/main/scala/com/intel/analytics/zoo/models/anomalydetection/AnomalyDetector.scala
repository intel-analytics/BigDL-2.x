package com.intel.analytics.zoo.models.anomalydetection

import com.intel.analytics.bigdl.Criterion
import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.optim.{OptimMethod, ValidationMethod, ValidationResult}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Shape, T}
import com.intel.analytics.zoo.feature.text.TextSet
import com.intel.analytics.zoo.models.common.ZooModel
import com.intel.analytics.zoo.pipeline.api.keras.layers._
import com.intel.analytics.zoo.pipeline.api.keras.models.{KerasNet, Sequential}
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

class AnomalyDetector[T: ClassTag] private(
                                            val featureShape: Shape,
                                            val hiddenLayers: Array[Int] = Array(8, 32, 15),
                                            val dropouts: Array[Float] = Array(0.2f, 0.2f, 0.2f)
                                          )
                                          (implicit ev: TensorNumeric[T])
  extends ZooModel[Tensor[T], Tensor[T], T] {

  override def buildModel(): AbstractModule[Tensor[T], Tensor[T], T] = {

    val model = Sequential()
    model.add(InputLayer(inputShape = featureShape))

    for (i <- 0 to hiddenLayers.length - 1) {
      model.add(LSTM(hiddenLayers(i), returnSequences = true))
        .add(Dropout(dropouts(i)))
    }

    model.add(LSTM(hiddenLayers.last, returnSequences = false))
      .add(Dropout(dropouts.last))
      .add(Dense(outputDim = 1))

    model.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }

  def compile(
               optimizer: OptimMethod[T],
               loss: Criterion[T],
               metrics: List[ValidationMethod[T]] = null)(implicit ev: TensorNumeric[T]): Unit = {
    model.asInstanceOf[KerasNet[T]].compile(optimizer, loss, metrics)
  }

  def fit(
           x: RDD[Sample[T]],
           batchSize: Int,
           nbEpoch: Int,
           validationData: RDD[Sample[T]] = null)(implicit ev: TensorNumeric[T]): Unit = {
    model.asInstanceOf[KerasNet[T]].fit(x, batchSize, nbEpoch, validationData)
  }

  def evaluate(
                x: RDD[Sample[T]],
                batchSize: Int)
              (implicit ev: TensorNumeric[T]): Array[(ValidationResult, ValidationMethod[T])] = {
    model.asInstanceOf[KerasNet[T]].evaluate(x, batchSize)
  }

  def predict(
               x: RDD[Sample[T]],
               batchPerThread: Int): RDD[Activity] = {
    model.asInstanceOf[KerasNet[T]].predict(x, batchPerThread)
  }

}

case class FeatureLabelIndex[T: ClassTag] private(
                                                   feature: Array[Array[T]],
                                                   label: T, index: Long) {
  override def toString =
    "value: " + feature
      .map(x => x.mkString("|")).mkString(",") + " label:" + label + " index:" + index
}


object AnomalyDetector {
  /**
    * The factory method to create an anomaly detector for single time series
    *
    * @param featureShape The input shape of features.
    * @param hiddenLayers Units of hidden layers of LSTM.
    * @param dropouts     Fraction of the input units to drop out. Float between 0 and 1.
    * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
    */
  def apply[@specialized(Float, Double) T: ClassTag](
                                                      featureShape: Shape,
                                                      hiddenLayers: Array[Int] = Array(8, 32, 15),
                                                      dropouts: Array[Float] = Array(0.2f, 0.2f, 0.2f)
                                                    )(implicit ev: TensorNumeric[T]): AnomalyDetector[T] = {
    new AnomalyDetector[T](featureShape, hiddenLayers, dropouts).build()
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

  /**
    * Compare predictions and truth to detect anomalies by ranking the absolute differences。
    * Most distant values are considered as anomalies.
    *
    * @param yTruth          Truth to be compared
    * @param yPredict        Predictions
    * @param anomalyFraction Int. What percentage of the number of total values compared to be
    *                        considered as anomalies.
    */
  def detectAnomalies[T: ClassTag](yTruth: RDD[T],
                                   yPredict: RDD[T],
                                   anomalyFraction: Int = 5): RDD[(T, T, Any)] = {
    require(yTruth.count() == yPredict.count(), s"length of predictions and truth should match")
    val totalCount = yTruth.count()

    val threshold: Float = yTruth.zip(yPredict)
      .map(x => diff(x._1, x._2))
      .sortBy(x => -x)
      .take((totalCount * anomalyFraction.toFloat / 100).toInt)
      .min

    detectAnomalies(yTruth, yPredict, threshold)
  }

  /**
    * Compare predictions and truth to detect anomalies by ranking the absolute differences.
    * Most distant values are considered as anomalies.
    *
    * @param yTruth    Truth to be compared
    * @param yPredict  Predictions
    * @param threshold Float. The threshold of absolute difference, data points with a difference
    *                  above the threshold is considered as anomalies.
    */
  def detectAnomalies[T: ClassTag](yTruth: RDD[T],
                                   yPredict: RDD[T],
                                   threshold: Float): RDD[(T, T, Any)] = {
    require(yTruth.count() == yPredict.count(), s"length of predictions and truth should match")
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

  /** Unroll a rdd of arrays to prepare features and labels.
    *
    * @param dataRdd      data to be unrolled with a length.
    * @param unrollLength the length of precious values to predict future value.
    * @param predictStep  use precious values to predict future value, default is 1.
    *
    *                     a simple example
    *                     data: (1,2,3,4,5,6); unrollLength: 2, predictStep: 1
    *                     features, label, index
    *                     (1,2), 3, 0
    *                     (2,3), 4, 1
    *                     (3,4), 5, 2
    *                     (4,5), 6, 3
    * */
  def unroll[T: ClassTag](dataRdd: RDD[Array[T]],
                          unrollLength: Int,
                          predictStep: Int = 1
                         ): RDD[FeatureLabelIndex[T]] = {

    val n = dataRdd.count()
    val indexRdd: RDD[(Array[T], Long)] = dataRdd.zipWithIndex()

    val offset: Int = unrollLength - 1 + predictStep

    //RDD[index of record, label]
    val labelRdd: RDD[(Long, T)] = indexRdd
      .filter(x => (x._2 >= offset))
      .map(x => (x._2 - offset, x._1(0)))

    //RDD[index of record, feature]
    val featureRdd: RDD[(Long, Array[Array[T]])] = indexRdd
      .flatMap(x => {
        val pairs: Seq[(Long, List[(Array[T], Long)])] = if (x._2 < unrollLength) {
          (0L to x._2).map(index => (index, List(x)))
        } else {
          (x._2 - unrollLength + 1 to x._2).map(index => (index, List(x)))
        }
        pairs
      }).reduceByKey(_ ++ _)
      .filter(x => x._2.size == unrollLength && x._1 <= n - unrollLength - predictStep)
      .map(x => {
        val data: Array[Array[T]] = x._2.sortBy(y => y._2).map(x => x._1).toArray
        (x._1, data)
      })
      .sortBy(x => x._1)

    val featureLabelIndex = featureRdd.join(labelRdd)
      .sortBy(x => x._1)
      .map(x => FeatureLabelIndex(x._2._1, x._2._2, x._1))

    featureLabelIndex
  }

  def toSampleRdd[T: ClassTag](rdd: RDD[FeatureLabelIndex[T]])(implicit ev: TensorNumeric[T]) = {
    rdd.map(x => {
      val shape: Array[Int] = Array(x.feature.length, x.feature(0).length)
      val data = x.feature.flatten
      val feature = Tensor[T](data, shape)
      Sample[T](feature, x.label)
    })
  }

}

