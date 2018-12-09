package com.intel.analytics.zoo.apps.recommendation.interence


import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.zoo.apps.recommendation.inference.NueralCFModel
import com.intel.analytics.zoo.models.recommendation.NeuralCF

object SimpleDriver {

  def main(args: Array[String]): Unit = {

    val modelPath = System.getProperty("MODEL_PATH", "./models/ncf.bigdl")

    val rcm = new NueralCFModel()
    rcm.load(modelPath)

    val userIds = ( 1 to 10)
    val itemIds = ( 2 to 6)

    val userItemPair = userIds.flatMap( x=> itemIds.map(y => (x,y)))

    val userItemFeature = rcm.preProcess(userItemPair)

    userItemFeature.map(x=> {
      val r = rcm.predict(x._2)
      println(x._1 +":" + r)
    })
  }
}
