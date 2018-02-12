package com.intel.analytics.zoo.pipeline.utils

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.{Container, SpatialConvolution, Utils}
import com.intel.analytics.bigdl.tensor.Tensor
import org.apache.log4j.Logger


object ModuleUtil {

  private val logger = Logger.getLogger(getClass)
  val shareFinput = Tensor[Float](1)

  /**
   * share the storage of SpatialConvolution fInput
   * note that this sharing only works for Inference only
   *
   * @param model model to share
   */
  def shareMemory(model: Module[Float]): Unit = {
    logger.info(s"Share memory in ${model.getName()}")
    shareFInput(model, shareFinput)
  }

  private def shareFInput(module: Module[Float], shareFinput: Tensor[Float]): Unit = {
    Utils.getNamedModules(module)
    module match {
      case m: Container[_, _, Float] =>
        for (m <- module.asInstanceOf[Container[_, _, Float]].modules) {
          shareFInput(m, shareFinput)
        }
      case _ =>
        if (module.getClass.getName.endsWith("SpatialConvolution")) {
          module.asInstanceOf[SpatialConvolution[Float]].fInput.set(shareFinput)
        }
    }
  }
}
