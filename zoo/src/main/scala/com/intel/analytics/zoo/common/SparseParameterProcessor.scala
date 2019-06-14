package com.intel.analytics.bigdl.parameters

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.optim.DistriOptimizer.Cache
import com.intel.analytics.bigdl.optim.{Metrics, OptimMethod}
import com.intel.analytics.bigdl.tensor.{SparseTensorUtils, Tensor, sudoLookupTableSparse}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

// U is walk around for collectGlobalData[T] is not ClassTag
class SparseParameterProcessor[U: ClassTag](optimMethods: OptimMethod[U])(implicit ev: TensorNumeric[U])
  extends ParameterProcessor {
  var globalSparseG: Tensor[U] = null
  var globalW: Tensor[U] = null
  var bcGlobalW: Broadcast[Tensor[U]] = null

  override def collectGlobalData[T](models: RDD[Cache[T]],
                                    parameters: AllReduceParameter[T],
                                    metrics: Metrics,
                                    state: Table)(implicit ev2: TensorNumeric[T]) : Unit = {
    // 1. aggregate sparseG first in each node
    // 2. aggregate sparseG on driver side
    globalSparseG = models.mapPartitions(modelIter => {
      val cached = modelIter.next()
      val sparseG = cached.localModels.map(
        // TODO: remove asInstanceOf
        _.asInstanceOf[sudoLookupTableSparse[U]].getSparseParameters()._2)

      // aggregate sparseG first in each node
      if (sparseG.length > 1) {
        var res = sparseG.head
        var i = 1
        while (i < sparseG.length) {
          res = SparseTensorUtils.addSparseTensor[U](res, sparseG(i))
          i += 1
        }
        Iterator(res)
      } else Iterator(sparseG.head)
    }).reduce(SparseTensorUtils.addSparseTensor[U](_, _))

    //TODO: support cliping sparseG

    // update weight with global gradients in driver
    if (globalW == null) {
      globalW = models.take(1).head.localModels.head
        .asInstanceOf[sudoLookupTableSparse[U]].getSparseParameters()._1
    }
    globalW = optimMethods.optimize(_ => (ev.fromType(1.0f), globalSparseG), globalW)._1

    // update weight in the cluster
    val sc = models.sparkContext
    bcGlobalW = sc.broadcast(globalW)
//    models.mapPartitions(modelIter => {
//      val modelCache = modelIter.next()
//      modelCache.localModels.head
//        .asInstanceOf[sudoLookupTableSparse[U]].setSparseParameters(broadcastW.value, null)
//      Iterator.empty
//    }).count()
  }

  override def processParameters[T](parameters: AllReduceParameter[T],
                                    modelCache: Cache[T],
                                    state: Table)(implicit ev: TensorNumeric[T]): Unit = {
      modelCache.localModels.head
        .asInstanceOf[sudoLookupTableSparse[U]].setSparseParameters(bcGlobalW.value, null)
  }

  // TODO: support sparseG for local mode
  override def processParameters[T](model: Module[T],
    state: Table)(implicit ev: TensorNumeric[T]): Unit = {
  }

  def getSparseParameters(model: Module[U]): Unit = {
    // TODO: remove asInstanceOf
    model.asInstanceOf[sudoLookupTableSparse[U]].setSparseParameters(globalW, globalSparseG)
  }
}
