package com.intel.analytics.zoo.serving.baseline

import com.intel.analytics.zoo.serving.TestUtils
import com.intel.analytics.zoo.serving.utils.Supportive
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class Operations extends FlatSpec with Matchers with BeforeAndAfter with Supportive {
  "basic operations" should "work" in {
    // read file from zoo/src/test/resources/serving to String
    // this is a prepared json format input of DIEN recommendation model
    val string = TestUtils.getStrFromResourceFile("dien_json_str_bs16.json")

    // decode json string input to activity
    val input =
  }
}
