package com.intel.analytics.zoo.pipeline.estimator

import com.intel.analytics.bigdl.{Criterion, DataSet, Module}
import com.intel.analytics.bigdl.dataset.{DistributedDataSet, LocalDataSet, MiniBatch}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.pipeline.api.keras.models.{InternalDistriOptimizer, InternalLocalOptimizer}

import scala.reflect.ClassTag

trait AbstractEstimator[T]{
  def train(trainSet: DataSet[MiniBatch[T]],
            optimMethod: OptimMethod[T] = null,
            endTrigger: Option[Trigger] = None,
            checkPoint: Option[Trigger] = None): this.type

  def evaluate(validationSet: DataSet[MiniBatch[T]],
               validationMethod: Array[ValidationMethod[T]]
              ): Map[ValidationMethod[T], ValidationResult]

}

class Estimator[T: ClassTag] private[zoo](
      model: Module[T],
      criterion: Criterion[T],
      modelDir: Option[String] = None)(implicit ev: TensorNumeric[T]) extends AbstractEstimator[T] {
  private var internalEstimator: AbstractEstimator[T] = null

  override def train(trainSet: DataSet[MiniBatch[T]],
            optimMethod: OptimMethod[T] = null,
            endTrigger: Option[Trigger] = None,
            checkPoint: Option[Trigger] = None): this.type = {
    if (internalEstimator != null) {
      internalEstimator.train(trainSet,
        optimMethod,
        endTrigger,
        checkPoint)
      this
    } else {
      val internalDistriOptimizer = trainSet match {
        case d: DistributedDataSet[MiniBatch[T]] =>
          new InternalDistriOptimizer[T](model, null, criterion)
        case l: LocalDataSet[MiniBatch[T]] =>
          new InternalLocalOptimizer[T](model, l, criterion)
      }
      internalDistriOptimizer.train(trainSet, optimMethod, endTrigger, checkPoint)
      this

    }
  }

  override def evaluate(validationSet: DataSet[MiniBatch[T]],
                        validationMethod: Array[ValidationMethod[T]]
              ): Map[ValidationMethod[T], ValidationResult] = {
    internalEstimator.evaluate(validationSet, validationMethod)
  }
}

object Estimator {
  // TODO: local or dist?
  def apply[T: ClassTag](
               model: Module[T],
               criterion: Criterion[T],
               modelDir: String = "")(implicit ev: TensorNumeric[T]): AbstractEstimator[T] = {
    val estimator = new InternalDistriOptimizer[T](model, null, criterion)
    if (modelDir != null && modelDir != "") {
      estimator.setCheckpointDir(modelDir)
    }
    estimator
  }

}