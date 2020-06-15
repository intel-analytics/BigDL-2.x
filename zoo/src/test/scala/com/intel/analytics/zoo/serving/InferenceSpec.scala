package com.intel.analytics.zoo.serving

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.serving.utils.{ClusterServingHelper, SerParams}
import org.scalatest.{FlatSpec, Matchers}

class InferenceSpec extends FlatSpec with Matchers {
  "TF String input" should "work" in {
//    val configPath = "/home/litchy/pro/analytics-zoo/config.yaml"
    val str = "abc|dff|aoa"
    val eleList = str.split("\\|")
//    val helper = new ClusterServingHelper(configPath)
//    helper.initArgs()
//    val param = new SerParams(helper)
//    val model = helper.loadInferenceModel()
//    val res = model.doPredict(t)
//    res
  }

}
