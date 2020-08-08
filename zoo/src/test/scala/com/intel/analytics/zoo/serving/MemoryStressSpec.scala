package com.intel.analytics.zoo.serving


import com.intel.analytics.zoo.serving.http.{Instances, JsonUtil}
import org.scalatest.{FlatSpec, Matchers}

class MemoryStressSpec extends FlatSpec with Matchers {
  "Stress" should "work" in {
    val inputStr = """{
                     |"instances": [
                     |   {
                     |     "t": [1, 2]
                     |   }
                     |]
                     |}
                     |""".stripMargin
    (0 until 10000000).foreach(i => {
      val ins = JsonUtil.fromJson(classOf[Instances], inputStr)
      val bytes = ins.toArrow()
      val b64 = java.util.Base64.getEncoder.encodeToString(bytes)
      if (i % 100000 == 0) {
        println(s"$i record to arrow completed. result $b64")
      }
    })
  }

}

