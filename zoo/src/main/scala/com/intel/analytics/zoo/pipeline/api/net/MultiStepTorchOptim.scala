package com.intel.analytics.zoo.pipeline.api.net

import com.intel.analytics.bigdl.optim.OptimMethod
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table
import com.intel.analytics.zoo.pipeline.api.keras.models.InternalOptimizerUtil

import scala.reflect.ClassTag

class MultiStepTorchOptim[@specialized(Float, Double) T: ClassTag](
    torchOptims: Array[TorchOptim[T]],
    endEpochs: Array[Int])(implicit ev: TensorNumeric[T]) extends OptimMethod[T] {
  require(torchOptims.length == endEpochs.length || torchOptims.length == endEpochs.length + 1,
    s"detect you have ${torchOptims.length} optimizers, but only ${endEpochs.length} endEpochs" +
      s", expected ${torchOptims.length} or ${torchOptims.length - 1} endEpochs.")

  override def optimize(
      feval: Tensor[T] => (T, Tensor[T]),
      parameter: Tensor[T]): (Tensor[T], Array[T]) = {
    val currentEpoch = TorchOptim.getEpoch(this)
    var currentOptim = 0
    var optimEpoch = currentEpoch
    for (i <- torchOptims.indices) {
      if (i >= endEpochs.length){
        currentOptim = i
        optimEpoch = currentEpoch - endEpochs.last
      } else if (currentEpoch >= endEpochs(i)) {
        currentOptim = i + 1
        optimEpoch = currentEpoch - endEpochs(i - 1)
      }
    }
    InternalOptimizerUtil.getStateFromOptiMethod(torchOptims(currentOptim))
      .update("Epoch", optimEpoch)
    torchOptims(currentOptim).optimize(feval, parameter)
  }

  override def clearHistory(): Unit = {

  }

  override def getLearningRate(): Double = {
    val currentEpoch = TorchOptim.getEpoch(this)
    var currentOptim = 0
    for (i <- torchOptims.indices) {
      if (i >= endEpochs.length){
        currentOptim = i
      } else if(currentEpoch >= endEpochs(i)) {
        currentOptim = i + 1
      }
    }
    torchOptims(currentOptim).getLearningRate()
  }

  override def loadFromTable(config: Table): this.type = {
    this
  }

  override def updateHyperParameter(): Unit = {
  }

  override def getHyperParameter(): String = {
    val currentEpoch = TorchOptim.getEpoch(this)
    var currentOptim = 0
    for (i <- torchOptims.indices) {
      if (i >= endEpochs.length){
        currentOptim = i
      } else if(currentEpoch >= endEpochs(i)) {
        currentOptim = i + 1
      }
    }
    torchOptims(currentOptim).getHyperParameter()
  }
}


object MultiStepTorchOptim{
  def apply[T: ClassTag](
      torchBytes: Array[Array[Byte]],
      epochs: Array[Int])(implicit ev: TensorNumeric[T]): MultiStepTorchOptim[T] = {
    new MultiStepTorchOptim[T](torchBytes.map(TorchOptim[T]), epochs)
  }
}
