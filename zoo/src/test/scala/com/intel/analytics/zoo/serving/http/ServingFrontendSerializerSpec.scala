package com.intel.analytics.zoo.serving.http

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.databind.module.SimpleModule
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.zoo.serving.TestUtils
import com.intel.analytics.zoo.serving.utils.ConfigParser
import org.scalatest.{FlatSpec, Matchers}

class ServingFrontendSerializerSpec extends FlatSpec with Matchers with Supportive {
  val configPath = getClass.getClassLoader.getResource("serving").getPath + "/config-test.yaml"

  val configParser = new ConfigParser(configPath)
  "read json string" should "work" in {
    val mapper = new ObjectMapper()
    val module = new SimpleModule()
    module.addDeserializer(classOf[Activity], new ServingFrontendSerializer())
    mapper.registerModule(module)

    val jsonStr = """{
  "instances" : [ {
    "intTensor" : [ 7756, 9549, 1094, 9808, 4959, 3831, 3926, 6578, 1870, 1741 ],
    "floatTensor" : [ 0.6804766, 0.30136853, 0.17394465, 0.44770062, 0.20275897 ],
    "intTensor2" : [ [ 1, 2 ], [ 3, 4 ], [ 5, 6 ] ],
    "floatTensor2" : [ [ [ 0.2, 0.3 ], [ 0.5, 0.6 ] ], [ [ 0.2, 0.3 ], [ 0.5, 0.6 ] ] ],
    "intScalar" : 12345,
    "floatScalar" : 3.14159
  } ]
}"""
    val tinyJsonStr = """{
  "instances" : [ {
    "intScalar" : 12345
  } ]
}"""

    timing("decode")() {
      (0 to 100).foreach(_ => {

        timing("decode once")() {
          mapper.readValue(jsonStr, classOf[Activity])
        }
        timing("decode tiny string")() {
          mapper.readValue(tinyJsonStr, classOf[Activity])
        }
      })

    }
    val a = mapper.readValue(jsonStr, classOf[Activity])
    val b = mapper.readValue(tinyJsonStr, classOf[Activity])
    a
  }
  "read dien string" should "work" in {
    val mapper = new ObjectMapper()
    val module = new SimpleModule()
    module.addDeserializer(classOf[Activity], new ServingFrontendSerializer())
    mapper.registerModule(module)
    val jsonStr = TestUtils.getStrFromResourceFile("dien_json_str.json")
    val a = mapper.readValue(jsonStr, classOf[Activity])
    a
  }
}

