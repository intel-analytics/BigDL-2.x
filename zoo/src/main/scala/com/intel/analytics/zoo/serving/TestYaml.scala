package com.intel.analytics.zoo.serving

import java.io.{File, FileInputStream}

import org.yaml.snakeyaml.Yaml


object TestYaml {
  def main(args: Array[String]): Unit = {
    val yaml = new Yaml()
    val input = new FileInputStream(new File("/home/litchy/pro/analytics-zoo/" +
      "zoo/src/main/scala/com/intel/analytics/zoo/serving/config.yaml"))
    val list = yaml.load(input).asInstanceOf[java.util.LinkedHashMap[String, String]]
    val s = list.get("spark").asInstanceOf[java.util.LinkedHashMap[String, String]]
    None
  }
}
