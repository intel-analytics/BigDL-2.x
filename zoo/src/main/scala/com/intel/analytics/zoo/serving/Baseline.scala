package com.intel.analytics.zoo.serving

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.serving.utils.ClusterServingHelper
import org.apache.log4j.{Level, Logger}

object Baseline {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.zoo.feature.image").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.zoo").setLevel(Level.INFO)
  def main(args: Array[String]): Unit = {
    val helper = new ClusterServingHelper()

    helper.initArgs()
    helper.initContext()
    val model = helper.loadInferenceModel()
    val bcModel = helper.sc.broadcast(model)

    val x = Tensor[Float](3 ,224, 224).rand()

    val arr = new Array[Tensor[Float]](10000)
    (0 until arr.size).foreach(i => arr(i) = x)
    val modelType = helper.modelType
    val batchSize = helper.batchSize

    val r = helper.sc.parallelize(arr, 1)
    val s = System.nanoTime()
    r.mapPartitions( a => {
      a.grouped(batchSize).flatMap(b => {
        val size = b.size

        val cpStart = System.nanoTime()
        val c = Tensor[Float](batchSize, 3, 224, 224)
        (0 until size).indices.toParArray.map(idx => {
          c.select(1, idx+1).copy(b(idx))

        })
        val x = if (modelType == "tensorflow") {
          c.transpose(2, 3)
            .transpose(3, 4).contiguous()
        } else if (modelType == "openvino") {
          c.addSingletonDimension()
        } else {
          c
        }
        val cpEnd = System.nanoTime()
        val localModel = bcModel.value

        val tkEnd = System.nanoTime()
        val cpE = (cpEnd - cpStart) / 1e9
        val tkE = (tkEnd - cpEnd) / 1e9
        println("cp time --> " + cpE.toString + "  take time -->" + tkE.toString)

        localModel.doPredict(x)
        (0 until 1)
      })
    }).count()

    val e = System.nanoTime()
    val elp = (e - s)/1e9
    println("10000 images " + elp)
  }

}
