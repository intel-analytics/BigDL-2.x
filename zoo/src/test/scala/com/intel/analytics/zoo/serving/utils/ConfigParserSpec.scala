package com.intel.analytics.zoo.serving.utils

import org.scalatest.{FlatSpec, Matchers}

class ConfigParserSpec extends FlatSpec with Matchers {
  val configPath = getClass.getClassLoader.getResource("serving").getPath + "/config-test.yaml"

  val configParser = new ConfigParser(configPath)
  "load config" should "work" in {
    val conf = configParser.loadConfig()
    assert(conf.modelPath.isInstanceOf[String])
    assert(conf.inputAlreadyBatched.isInstanceOf[Boolean])
    assert(conf.redisSecureStructStorePassword.isInstanceOf[String])
  }

}
