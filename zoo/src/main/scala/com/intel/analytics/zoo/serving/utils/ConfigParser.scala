package com.intel.analytics.zoo.serving.utils

import java.io.{File, FileInputStream}

import org.yaml.snakeyaml.Yaml
import org.yaml.snakeyaml.constructor.Constructor

import scala.beans.BeanProperty

class ConfigParser(configPath: String) {

  def loadConfig(): ClusterServingHelper = {
    val yamlParser = new Yaml(new Constructor(classOf[ClusterServingHelper]))
    val input = new FileInputStream(new File(configPath))
    try {
      val helper = yamlParser.load(input).asInstanceOf[ClusterServingHelper]
      parseConfigStrings(helper)
    }
    catch {
      case e: Exception =>
        println(s"Invalid configuration, please check type regulations in config file")
        e.printStackTrace()
        throw new Error("Configuration parsing error")
    }
  }
  def parseConfigStrings(clusterServingHelper: ClusterServingHelper): ClusterServingHelper = {
    clusterServingHelper.redisHost = clusterServingHelper.redisUrl.split(":").head.trim
    clusterServingHelper.redisPort = clusterServingHelper.redisUrl.split(":").last.trim.toInt
    clusterServingHelper
  }
}