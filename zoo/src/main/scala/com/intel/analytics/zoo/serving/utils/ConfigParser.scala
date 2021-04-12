/*
 * Copyright 2018 Analytics Zoo Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.zoo.serving.utils

import java.io.{File, FileInputStream}

import org.yaml.snakeyaml.Yaml
import org.yaml.snakeyaml.constructor.Constructor


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
