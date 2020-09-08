package com.intel.analytics.zoo.serving.operator

class ClusterServingParams(modelPath: String,
                           modelConcurrent: Int = 1,
                           inferenceMode: String = "single") extends Serializable {
  val _modelPath = modelPath
  val _modelConcurrent = modelConcurrent
  val _inferenceMode = inferenceMode
}
