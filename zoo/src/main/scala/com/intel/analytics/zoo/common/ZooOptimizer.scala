package com.intel.analytics.zoo.common

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.{DistributedDataSet, MiniBatch}
import com.intel.analytics.bigdl.optim.{DistriOptimizer, OptimMethod}
import com.intel.analytics.bigdl.parameters.SparseParameterProcessor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class ZooOptimizer[T: ClassTag] (
     _model: Module[T],
     _dataset: DistributedDataSet[MiniBatch[T]],
     _criterion: Criterion[T]
   )(implicit ev: TensorNumeric[T])
  extends DistriOptimizer[T](
    _model, _dataset, _criterion) {
  var sparseParameterProcessor: SparseParameterProcessor[T] = null

  def setSparseParameterProcessor(optimMethods: OptimMethod[T]): this.type = {
    sparseParameterProcessor = new SparseParameterProcessor[T](optimMethods)
    parameterProcessors.append(sparseParameterProcessor)
    this
  }

  override def optimize(): Module[T] = {
    val model = super.optimize()
    sparseParameterProcessor.getSparseParameters(model)
    model
  }
}
