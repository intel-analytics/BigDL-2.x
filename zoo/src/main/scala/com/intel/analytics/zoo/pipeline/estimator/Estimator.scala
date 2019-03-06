package com.intel.analytics.zoo.pipeline.estimator

import com.intel.analytics.bigdl.{Criterion, DataSet, Module}
import com.intel.analytics.bigdl.dataset.MiniBatch
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.pipeline.api.keras.models.InternalDistriOptimizer

import scala.collection.immutable.HashMap
import scala.reflect.ClassTag

trait Estimator[T]{
  def train(trainSet: DataSet[MiniBatch[T]],
            optimMethod: OptimMethod[T] = null,
            trigger: Option[Trigger] = None,
            steps: Option[Int] = None,
            maxSteps: Option[Int] = None,
            checkPoint: Option[Trigger] = None): this.type

  def evaluate(validationSet: DataSet[MiniBatch[T]]
              ): HashMap[ValidationMethod[T], ValidationResult]

}

object Estimator {
  // TODO: local or dist?
  def apply[T: ClassTag](
               model: Module[T],
               criterion: Criterion[T],
               modelDir: Option[String] = None)(implicit ev: TensorNumeric[T]): Estimator[T] = {
    val estimator = new InternalDistriOptimizer[T](model, null, criterion)
    if (modelDir.isDefined) {
      estimator.setCheckpointDir(modelDir.get)
    }
    estimator
  }

}