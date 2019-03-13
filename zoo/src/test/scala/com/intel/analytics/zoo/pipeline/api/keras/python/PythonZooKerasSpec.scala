package com.intel.analytics.zoo.pipeline.api.keras.python

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.python.api.PythonBigDL
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.pipeline.api.keras.layers.{Dense, Flatten}
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}
import com.intel.analytics.zoo.pipeline.api.keras.python.PythonZooKeras

class PythonZooKerasSpec extends FlatSpec with Matchers with BeforeAndAfter {
  private var sc: SparkContext = _

  def generateData(featureShape: Array[Int], labelSize: Int, dataSize: Int): RDD[Sample[Float]] = {
    sc.range(0, dataSize, 1).map { _ =>
      val featureTensor = Tensor[Float](featureShape)
      featureTensor.apply1(_ => scala.util.Random.nextFloat())
      val labelTensor = Tensor[Float](labelSize)
      labelTensor(Array(labelSize)) = Math.round(scala.util.Random.nextFloat())
      Sample[Float](featureTensor, labelTensor)
    }
  }
  before {
    val conf = new SparkConf()
      .setMaster("local[4]")
    sc = NNContext.initNNContext(conf, appName = "PythonZooKerasSpec")
  }

  after {
    if (sc != null) {
      sc.stop()
    }
  }


  "zooEvaluate" should "work" in {
    val trainingData = generateData(Array(12, 12), 1, 100)
    val model = Sequential[Float]()

    model.add(Dense[Float](8, activation = "relu", inputShape = Shape(12, 12)))
    model.add(Flatten[Float]())
    model.add(Dense[Float](2, activation = "softmax"))

    model.compile(optimizer = "sgd", loss = "sparse_categorical_crossentropy",
      metrics = List("accuracy"))
    model.fit(trainingData, batchSize = 8, nbEpoch = 2)

    val api = new PythonZooKeras[Float]()
    val bigdlApi = sc.broadcast(new PythonBigDL[Float]())

    /** python api require to take no type Sample
      * and it takes JavaRDD as input
      *
      * use toPySample to convert to no type
      * use toJavaRDD to convert to JavaRDD
    **/
    val jd = trainingData.map(j => bigdlApi.value.toPySample(j)).toJavaRDD()
    val res = api.zooEvaluate(model, jd, 8)
    res
  }
}
