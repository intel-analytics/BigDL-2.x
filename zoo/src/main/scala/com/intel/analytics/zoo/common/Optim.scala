package com.intel.analytics.zoo.common

import com.intel.analytics.bigdl.optim.SGD
import com.intel.analytics.bigdl.optim.SGD.LearningRateSchedule
import com.intel.analytics.bigdl.utils.Table
import com.intel.analytics.zoo.pipeline.api.keras.models.{InternalOptimizerUtil}

object Optim {

  /**
   * A fixed learning rate scheduler, always return the same learning rate
   * @param lr learning rate
   */
  case class Fixed(lr: Double) extends LearningRateSchedule {
    override def updateHyperParameter(config: Table, state: Table): Unit = {
      val nevals = state.get[Int]("evalCounter").getOrElse(0)
      state("evalCounter") = nevals + 1
      config("clr") = lr
    }

    override def updateHyperParameter[T](optimMethod: SGD[T]): Unit = {
      val state = InternalOptimizerUtil.getStateFromOptiMethod[T](optimMethod)
      val nevals = state.get[Int]("evalCounter").getOrElse(0)
      state("evalCounter") = nevals + 1
      currentRate = lr
    }
  }
}
