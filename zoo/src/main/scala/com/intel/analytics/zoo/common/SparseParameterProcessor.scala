package com.intel.analytics.bigdl.parameters

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.abstractnn.SparseAbstractModule
import com.intel.analytics.bigdl.optim.DistriOptimizer.Cache
import com.intel.analytics.bigdl.optim.{Metrics, OptimMethod}
import com.intel.analytics.bigdl.tensor.{IndexedSlicesTensor, SparseTensorUtils, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table
import com.intel.analytics.zoo.pipeline.api.keras.optimizers.SparseOptimMethod
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

// U is walk around for collectGlobalData[T] is not ClassTag
class SparseParameterProcessor[U: ClassTag](optimMethods: SparseOptimMethod[U])
  (implicit ev: TensorNumeric[U]) extends ParameterProcessor {
//  var globalSparseG: Array[Tensor[U]] = null
//  var globalW: Array[Tensor[U]] = null
//  var bcGlobalW: Broadcast[Array[Tensor[U]]] = null

  override def collectGlobalData[T](models: RDD[Cache[T]],
                                    parameters: AllReduceParameter[T],
                                    metrics: Metrics,
                                    state: Table)(implicit ev2: TensorNumeric[T]) : Unit = {
    // 1. aggregate sparseG first in each node
    // 2. aggregate sparseG on driver side
    val globalSparseG = models.mapPartitions(modelIter => {
      val cached = modelIter.next()

      val sparseG = cached.localModels.map(
        _.asInstanceOf[SparseAbstractModule[U]].sparseParameters()._2)

      // aggregate sparseG first in each node
      if (sparseG.length > 1) {
        var aggregatedG = sparseG.head
        var i = 1
        while (i < sparseG.length) {

          aggregatedG = SparseTensorUtils.addSparseTensor[U](aggregatedG, sparseG(i))
          i += 1
        }
        Iterator(aggregatedG)
      } else Iterator(sparseG.head)
    }).reduce(SparseTensorUtils.addSparseTensor[U](_, _))

    globalSparseG.foreach {x =>
      SparseTensorUtils.dotSparseTensorValueByConstant[U](x,
        ev.fromType(1.0 / state[Int]("numFinishedModel")))}

//    val t = Array[Float](0.0025634766f,0.0053710938f,0.0067443848f,10.9375f,22.875f,28.875f,
//      0.020019531f,0.041748047f,0.052734375f,0.008361816f,0.017456055f,0.021972656f,-10.9375f,
//      -22.875f,-28.875f)
//    SparseTensorUtils.updateValues(globalSparseG.head, t.asInstanceOf[Array[U]])

    //TODO: support cliping sparseG

    // update weight with global gradients in driver
    var globalW =
//    if (bcGlobalW == null) {
      models.take(1).head.localModels.head
        .asInstanceOf[SparseAbstractModule[U]].sparseParameters()._1
//    } else bcGlobalW.value
//    globalW = optimMethods.optimize(_ => (ev.fromType(1.0f), globalSparseG), globalW)._1

//    globalW = globalW.zip(globalSparseG).map { case(w, g) =>
//      optimMethods.optimize(_ => (ev.fromType(1.0f), g), w)._1}
    val value = Array.fill(globalW.length)(ev.fromType((1.0f)))
    globalW = optimMethods.optimize2(_ => (value, globalSparseG), globalW)._1

    // update weight in the cluster
    val sc = models.sparkContext
    val bcGlobalW = sc.broadcast(globalW)
    models.mapPartitions(modelIter => {
      val modelCache = modelIter.next()
      modelCache.localModels.head
        .asInstanceOf[SparseAbstractModule[U]].sparseParameters()._1.zip(bcGlobalW.value)
        .foreach { case (w, bw) => w.copy(bw)}
      Iterator.empty
    }).count()
  }

  override def processParameters[T](parameters: AllReduceParameter[T],
                                    modelCache: Cache[T],
                                    state: Table)(implicit ev: TensorNumeric[T]): Unit = {
//      modelCache.localModels.head
//        .asInstanceOf[SparseAbstractModule[U]].sparseParameters()._1.zip(bcGlobalW.value)
//        .foreach {case (w, bw) => w.copy(bw)}
  }

  // TODO: support sparseG for local mode
  override def processParameters[T](model: Module[T],
    state: Table)(implicit ev: TensorNumeric[T]): Unit = {
  }

  def getSparseParameters(model: Module[U]): Unit = {
    // TODO: get sparseG and w
    model.asInstanceOf[SparseAbstractModule[U]].setSparseParameters(null, null)
  }
}
