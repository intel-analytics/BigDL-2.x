//package com.intel.analytics.zoo.optim
//
//import com.intel.analytics.bigdl.dataset.{MiniBatch, Sample}
//import org.apache.spark.rdd.RDD
//import com.intel.analytics.bigdl.optim.{Adam, Trigger}
//
//import com.intel.analytics.bigdl.optim.{Optimizer => BDOptmizer}
//import com.intel.analytics.zoo.models.common.ZooModel
//import com.intel.analytics.zoo.models.recommendation.Recommender
//
//class Optimizer
//
//object Optimizer {
//
//  def splitRdds(fullRdd: RDD[Sample[Float]], nSplits: Int = 2): RDD[(Int, Sample[Float])] = {
//
//    val rddSplited = fullRdd.map(x => {
//      val hashcode = x.hashCode()
//      val group = if (hashcode < 0) {
//        hashcode % nSplits + nSplits
//      } else {
//        hashcode % nSplits
//      }
//      (group, x)
//    })
//    rddSplited
//  }
//
//  def optimize(optimizer: BDOptmizer[Float, MiniBatch[Float]],
//              model: Recommender[Float],
//              fullRdd: RDD[Sample[Float]],
//              batchSize: Int,
//              optimMethod: Adam[Float],
//              nSplits: Int = 2) = {
//
//    val rddSplited = splitRdds(fullRdd, nSplits)
//
//    (0 to nSplits - 1).map(i => {
//      val subTrainRdd = rddSplited.filter(x => x._1 == i).map(x => x._2)
//      optimizer
//        .setModel(model)
//        .setOptimMethod(optimMethod)
//        .setEndWhen(Trigger.maxEpoch(10))
//        .setTrainData(subTrainRdd, batchSize = batchSize)
//        .optimize()
//    })
//  }
//
//}
