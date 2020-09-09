package com.intel.analytics.zoo.serving.operator

class ClusterServingParams(modelPath: String,
                           modelConcurrent: Int = 1,
                           inferenceMode: String = "single",
                           coreNum: Int = 4) extends Serializable {
  val _modelPath = modelPath
  val _modelConcurrent = modelConcurrent
  val _inferenceMode = inferenceMode
  val _coreNum = coreNum
  var _modelType: String = _
}
