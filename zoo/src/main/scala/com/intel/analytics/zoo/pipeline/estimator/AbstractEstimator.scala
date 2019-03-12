package com.intel.analytics.zoo.pipeline.estimator

import com.intel.analytics.bigdl.{Criterion, DataSet, Module}
import com.intel.analytics.bigdl.dataset.{DistributedDataSet, LocalDataSet, MiniBatch}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.pipeline.api.keras.models.{InternalDistriOptimizer, InternalLocalOptimizer}

import scala.reflect.ClassTag

trait AbstractEstimator[T]{
  def train(trainSet: DataSet[MiniBatch[T]],
            criterion: Criterion[T] = null,
            endTrigger: Option[Trigger] = None,
            checkPoint: Option[Trigger] = None,
            validationSet: DataSet[MiniBatch[T]] = null,
            validationMethod: Array[ValidationMethod[T]] = null): this.type

  def evaluate(validationSet: DataSet[MiniBatch[T]],
               validationMethod: Array[ValidationMethod[T]]
              ): Map[ValidationMethod[T], ValidationResult]

}

class Estimator[T: ClassTag] private[zoo](
      model: Module[T],
      optimMethods: Map[String, OptimMethod[T]] = Map(),
      modelDir: Option[String] = None)(implicit ev: TensorNumeric[T]) extends AbstractEstimator[T] {
  protected var internalEstimator: AbstractEstimator[T] = null

  override def train(trainSet: DataSet[MiniBatch[T]],
            criterion: Criterion[T],
            endTrigger: Option[Trigger] = None,
            checkPoint: Option[Trigger] = None,
            validationSet: DataSet[MiniBatch[T]] = null,
            validationMethod: Array[ValidationMethod[T]] = null): this.type = {
    if (internalEstimator == null) {
      internalEstimator = trainSet match {
        case d: DistributedDataSet[MiniBatch[T]] =>
          new InternalDistriOptimizer[T](model, null, criterion)
            .setCheckpointDir(modelDir)
            .setOptimMethods(optimMethods)
        case l: LocalDataSet[MiniBatch[T]] =>
          new InternalLocalOptimizer[T](model, l, criterion)
          // TODO
      }
    }
    internalEstimator.train(trainSet, criterion, endTrigger, checkPoint,
      validationSet, validationMethod)
    this
  }

  override def evaluate(validationSet: DataSet[MiniBatch[T]],
                        validationMethod: Array[ValidationMethod[T]]
              ): Map[ValidationMethod[T], ValidationResult] = {
    if (internalEstimator == null) {
      internalEstimator = validationSet match {
        case d: DistributedDataSet[MiniBatch[T]] =>
          new InternalDistriOptimizer[T](model, null, null)
            .setCheckpointDir(modelDir)
            .setOptimMethods(optimMethods)
        case l: LocalDataSet[MiniBatch[T]] =>
          new InternalLocalOptimizer[T](model, l, null)
        // TODO
      }
    }
    internalEstimator.evaluate(validationSet, validationMethod)
  }
}

object Estimator {
  // TODO: local or dist?
  def apply[T: ClassTag](
        model: Module[T],
        optimMethods: Map[String, OptimMethod[T]],
        modelDir: String)(implicit ev: TensorNumeric[T]): AbstractEstimator[T] = {
    if (null != modelDir && "" != modelDir) {
      new Estimator[T](model, optimMethods, Some(modelDir))
    } else {
      new Estimator[T](model, optimMethods)
    }
  }

  def apply[T: ClassTag](
       model: Module[T],
       optimMethods: Map[String, OptimMethod[T]]
      )(implicit ev: TensorNumeric[T]): AbstractEstimator[T] = {
    apply(model, optimMethods, "")
  }

  def apply[T: ClassTag](
        model: Module[T],
        optimMethod: OptimMethod[T],
        modelDir: String)(implicit ev: TensorNumeric[T]): AbstractEstimator[T] = {
    if (null != modelDir && "" != modelDir) {
      new Estimator[T](model, Map(model.getName() -> optimMethod), Some(modelDir))
    } else {
      new Estimator[T](model, Map(model.getName() -> optimMethod))
    }
  }

  def apply[T: ClassTag](
        model: Module[T],
        optimMethod: OptimMethod[T])(implicit ev: TensorNumeric[T]): AbstractEstimator[T] = {
    apply(model, optimMethod, "")
  }

  def apply[T: ClassTag](
        model: Module[T])(implicit ev: TensorNumeric[T]): AbstractEstimator[T] = {
    new Estimator[T](model, Map())
  }

}