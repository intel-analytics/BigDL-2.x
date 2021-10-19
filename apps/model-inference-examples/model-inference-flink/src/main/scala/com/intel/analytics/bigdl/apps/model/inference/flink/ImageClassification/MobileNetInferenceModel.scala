package com.intel.analytics.bigdl.apps.model.inference.flink.ImageClassification

import com.intel.analytics.bigdl.serving.pipeline.inference.InferenceModel

class MobileNetInferenceModel(var concurrentNum: Int = 1, modelPath: String, modelType: String, inputs: Array[String], outputs: Array[String], intraOpParallelismThreads: Int, interOpParallelismThreads: Int, usePerSessionThreads: Boolean) extends InferenceModel(concurrentNum) with Serializable {
  doLoadTensorflow(modelPath, modelType, inputs, outputs, intraOpParallelismThreads, interOpParallelismThreads, usePerSessionThreads)
}
