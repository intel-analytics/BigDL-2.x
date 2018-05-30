package com.intel.analytics.zoo.pipeline.inference

import com.intel.analytics.bigdl.optim.LocalPredictor
import com.intel.analytics.bigdl.nn.Module
import org.slf4j.LoggerFactory

object InferenceModelFactory extends InferenceSupportive{
  override val logger = LoggerFactory.getLogger(getClass)

  def loadFloatInferenceModel(modelPath: String): FloatInferenceModel = {
    loadFloatInferenceModel(modelPath, null)
  }

  def loadFloatInferenceModel(modelPath: String, weightPath: String): FloatInferenceModel = {
    timing(s"load FloatInferenceModel") {
      logger.info(s"load model from $modelPath and $weightPath")
      val model = Module.loadModule[Float](modelPath, weightPath)
      logger.info(s"loaded model as $model")
      val predictor = LocalPredictor(model)
      model.evaluate()
      FloatInferenceModel(model, predictor)
    }
  }
}