package com.intel.analytics.zoo.serving.utils

import org.apache.log4j.Logger

class SerParams(helper: ClusterServingHelper) extends Serializable {
  val redisHost = helper.redisHost
  val redisPort = helper.redisPort.toInt
  val coreNum = helper.coreNum
  val filter = helper.filter
  val chwFlag = helper.chwFlag
  val C = helper.dataShape(0)
  val H = helper.dataShape(1)
  val W = helper.dataShape(2)
  val modelType = helper.modelType
  val model = helper.loadInferenceModel()
}
