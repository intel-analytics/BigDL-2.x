package com.intel.analytics.zoo.serving.http

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.databind.module.SimpleModule
import com.intel.analytics.zoo.serving.utils.ConfigParser
import org.scalatest.{FlatSpec, Matchers}

class ServingFrontendSerializerSpec extends FlatSpec with Matchers {
  val configPath = getClass.getClassLoader.getResource("serving").getPath + "/config-test.yaml"

  val configParser = new ConfigParser(configPath)
  "read json string" should "work" in {
    val mapper = new ObjectMapper()
    val module = new SimpleModule()
    module.addDeserializer(classOf[Instances], new ServingFrontendSerializer())
    mapper.registerModule(module)

    val jsonStr = """{
  "instances" : [ {
    "intScalar" : 12345,
    "floatScalar" : 3.14159,
    "stringScalar" : "hello, world. hello, zoo.",
    "intTensor" : [ 7756, 9549, 1094, 9808, 4959, 3831, 3926, 6578, 1870, 1741 ],
    "floatTensor" : [ 0.6804766, 0.30136853, 0.17394465, 0.44770062, 0.20275897 ],
    "stringTensor" : [ "come", "on", "united" ],
    "intTensor2" : [ [ 1, 2 ], [ 3, 4 ], [ 5, 6 ] ],
    "floatTensor2" : [ [ [ 0.2, 0.3 ], [ 0.5, 0.6 ] ], [ [ 0.2, 0.3 ], [ 0.5, 0.6 ] ] ],
    "stringTensor2" : [ [ [ [ "come", "on", "united" ], [ "come", "on", "united" ] ] ] ],
    "sparseTensor" : {
      "shape" : [ 100, 10000, 10 ],
      "data" : [ 0.2, 0.5, 3.45, 6.78 ],
      "indices" : [ [ 1, 1, 1 ], [ 2, 2, 2 ], [ 3, 3, 3 ], [ 4, 4, 4 ] ]
    },
    "image": "/9j/4AAQSkZJRgABAQEASABIAAD/7RcEUGhvdG9za..."
  } ]
}"""
    val a = mapper.readValue(jsonStr, classOf[Instances])
  }
  "load default config" should "work" in {

  }
}

