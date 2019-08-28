package com.intel.analytics.bigdl.parameters

import java.util.concurrent.{Callable, Future}

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.abstractnn.SparseAbstractModule
import com.intel.analytics.bigdl.optim.DistriOptimizer.Cache
import com.intel.analytics.bigdl.optim.{Metrics, OptimMethod}
import com.intel.analytics.bigdl.tensor.{IndexedSlicesTensor, SparseTensorUtils, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Engine, Table}
import com.intel.analytics.zoo.common.SparseAllReduceParameter
import com.intel.analytics.zoo.pipeline.api.keras.optimizers.SparseOptimMethod
import org.apache.commons.lang.exception.ExceptionUtils
import org.apache.spark.TaskContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.sparkExtension.SparkExtension
import org.apache.spark.storage.{BlockId, BlockManagerWrapper, StorageLevel}

import scala.reflect._

// U is walk around for collectGlobalData[T] is not ClassTag
class SparseParameterProcessor2[U: ClassTag](weights: Array[Tensor[U]], partitionNum: Int, optimMethods: SparseOptimMethod[U])
  (implicit ev: TensorNumeric[U]) extends ParameterProcessor {

  // create SparseAllReduceParameter for each sparse layer
  val sparseAllReduceParameters = weights.map { weight =>
    new SparseAllReduceParameter[U](partitionNum, weight.size())
  }

  override def collectGlobalData[T](models: RDD[Cache[T]],
                                    parameters: AllReduceParameter[T],
                                    metrics: Metrics,
                                    state: Table)(implicit ev2: TensorNumeric[T]) : Unit = {
    // 1. aggregate sparseG first in each node
    models.mapPartitions(modelIter => {
      val cached = modelIter.next()

      val sparseG = cached.localModels.map(
        _.asInstanceOf[SparseAbstractModule[U]].sparseParameters()._2)

      // aggregate sparseG first in each node
      if (sparseG.length > 1) {
        var aggregatedG = sparseG.head
        var i = 1
        while (i < sparseG.length) {
          aggregatedG = SparseTensorUtils.addSparseTensor[U](aggregatedG, sparseG(i))
          // put gradients
          sparseAllReduceParameters(i).putGradients(aggregatedG(i))
          i += 1
        }
        Iterator.empty
      } else {
        // put gradients
        sparseAllReduceParameters.zip(sparseG.head).foreach {case (par, s) => par.putGradients(s)}
        Iterator.empty
      }
    }).count()
  }

  override def processParameters[T](parameters: AllReduceParameter[T],
                                    modelCache: Cache[T],
                                    state: Table)(implicit ev2: TensorNumeric[T]): Unit = {

    // aggregate part of gradients in each node
    sparseAllReduceParameters.foreach(_.aggregateGradientPartition(state[Int]("numFinishedModel")))

    val value = Array.fill(weights.length)(ev.fromType((1.0f)))

    optimMethods.optimize2(_ => (value, sparseAllReduceParameters.map(x => x.gradientPartition)),
      sparseAllReduceParameters.map(x => x.weightPartition))

    sparseAllReduceParameters.foreach(_.sendWeightPartition())

    sparseAllReduceParameters.foreach(_.getWeights(modelCache.modelWeights.head.asInstanceOf[Tensor[U]]))
  }

  // TODO: support sparseG for local mode
  override def processParameters[T](model: Module[T],
    state: Table)(implicit ev: TensorNumeric[T]): Unit = {
  }

  def getSparseParameters(model: Module[U]): Unit = {
    // TODO: get sparse weight, gradient
    model.asInstanceOf[SparseAbstractModule[U]].setSparseParameters(null, null)
  }
}
