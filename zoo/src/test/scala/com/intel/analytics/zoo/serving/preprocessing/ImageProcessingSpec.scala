package com.intel.analytics.zoo.serving.preprocessing

import com.intel.analytics.zoo.serving.utils.ConfigParser
import com.intel.analytics.zoo.serving.{ClusterServing, TestUtils}
import org.scalatest.{FlatSpec, Matchers}

class ImageProcessingSpec extends FlatSpec with Matchers {
  ClusterServing.helper = new ConfigParser(
    getClass().getClassLoader().getResource("serving").getPath
      + "/image-test-config.yaml").loadConfig()
  "image resize" should "work" in {
    val imageB64 = TestUtils.getStrFromResourceFile("image-3_224_224-jpg-base64")
    val preProcessing = new PreProcessing()
    val tensor = preProcessing.decodeImage(imageB64)
    assert(tensor.size()(0) == 3)
    assert(tensor.size()(1) == 225)
    assert(tensor.size()(2) == 225)
  }

}
