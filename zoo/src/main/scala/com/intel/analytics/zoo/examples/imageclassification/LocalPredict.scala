package com.intel.analytics.zoo.examples.imageclassification

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.feature.image._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.transform.vision.image.ImageFeature
import com.intel.analytics.zoo.feature.common.Preprocessing
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext

/**
  * Created by yuhao on 2/6/19.
  */
object LocalPredict {

  def main(args: Array[String]): Unit = {
    println(args.mkString("\n"))

    Logger.getLogger("org").setLevel(Level.WARN)
    val modelPath = args(0) // "/home/yuhao/workspace/model/analytics-zoo_resnet-50_imagenet_0.1.0.model"
    val imagePath = args(1) //"/home/yuhao/workspace/github/hhbyyh/Test/ZooExample/input"
    val mode = args(2) // local or distributed
    val partitions = args(3).toInt

    require(Seq("local", "distributed").contains(mode))

    val sc = NNContext.initNNContext("imageinfer")
    val transformer = ImageResize(256, 256) -> ImageCenterCrop(224, 224) ->
      ImageChannelNormalize(123, 117, 104) -> ImageMatToTensor() -> ImageSetToSample()
    val model = Module.loadModule(modelPath).evaluate()

    val repeat = 4 // repeat multi times to get throughput
    if (mode == "local") {
      localInference(model, transformer, imagePath) // warm up
      val avg = (1 to repeat).map(i => localInference(model, transformer, imagePath)).sum / repeat
      println(s"avg throughtput: $avg")
    } else {
      distributedInference(model, transformer, imagePath, sc, partitions) // warm up
      val avg = (1 to repeat).map(i => distributedInference(model, transformer, imagePath, sc, partitions)).sum / repeat
      println(s"avg throughtput: $avg")
    }
  }


  /**
    * read images from local file system and run inference locally without sparkcontext
    * master = local[x]
    * only support local file system
    */
  def localInference(
                      model: Module[Float],
                      transformer: Preprocessing[ImageFeature, ImageFeature],
                      imagePath: String): Double = {
    val st = System.nanoTime()
    val images = ImageSet.read(imagePath, imageCodec = 1)
    val features = images.transform(transformer)
    val result = model.predictImage(features.toImageFrame(), batchPerPartition = 4)
    val output = result.toLocal().array.head.predict()
    val inferTime = (System.nanoTime() - st) / 1e9
    val throughput = images.toLocal().array.length / inferTime
    println("\ninference takes: " + inferTime)
    println(s"throughput: $throughput" )
    throughput
  }

  /**
    * run inference in cluster mode, with spark overhead.
    * use master = local[x] or yarn
    * support HDFS path
    */
  def distributedInference(
                            model: Module[Float],
                            transformer: Preprocessing[ImageFeature, ImageFeature],
                            imagePath: String,
                            sc: SparkContext,
                            partitions: Int): Double = {
    val st = System.nanoTime()
    val images = ImageSet.read(imagePath, sc, imageCodec = 1).toDistributed().repartition(partitions)
//    images.toDistributed().rdd = images.toDistributed().rdd.repartition(partitions)
    println("#number of partitions: " + images.toDistributed().rdd.partitions.length)
    val features = images.transform(transformer)
    val result = model.predictImage(features.toImageFrame(), batchPerPartition = 8)
    val output = result.toDistributed().rdd.collect().head.predict()
    val inferTime = (System.nanoTime() - st)/1e9
    val throughput = images.toDistributed().rdd.count() / inferTime
    println("inference takes: " + inferTime)
    println("throughput: " + throughput)
    println()
    throughput
  }

}

// local[1] in local : 4.57
// local[1] in distributed: 4.15
// 1.0410767617646306

