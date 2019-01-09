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

package com.intel.analytics.zoo.pipeline.inference

import java.io.{File, FileOutputStream}
import java.nio.channels.Channels

import com.google.common.io.Files
import com.intel.analytics.zoo.pipeline.inference.DeviceType.DeviceTypeEnumVal
import org.slf4j.LoggerFactory

import scala.util.Random
import scala.language.postfixOps
import sys.process._

class OpenVinoInferenceSupportive {

  @native def loadOpenVinoIR(modelFilePath: String,
                             weightFilePath: String,
                             deviceTypeValue: Int): Long

  @native def predict(executableNetworkReference: Long,
                      data: Array[Float],
                      shape: Array[Int]): JTensor

  @native def releaseOpenVINOIR(executableNetworkReference: Long): Unit
}

object OpenVinoInferenceSupportive extends InferenceSupportive {
  val logger = LoggerFactory.getLogger(getClass)

  load("libinference_engine.so")
  load("libiomp5.so")
  load("libcpu_extension.so")
  load("libMKLDNNPlugin.so")
  load("libzoo_inference.so")

  def loadTensorflowModel(frozenModelFilePath: String,
                          pipelineConfigFilePath: String,
                          extensionsConfigFilePath: String,
                          deviceType: DeviceTypeEnumVal): OpenVinoInferenceModel = {
    logger.info(s"start to optimize tensorflow model from " +
      s"$frozenModelFilePath, $pipelineConfigFilePath, $extensionsConfigFilePath")
    timing("load tensorflow model to openvino IR") {
      val loadTensorflowModelScriptPath: String = OpenVinoInferenceSupportive.
        getClass.getResource("/zoo-openvino-mo-run.sh").getPath()
      val tmpDir = Files.createTempDir()
      val outputPath: String = tmpDir.getCanonicalPath

      val logFilePath = s"$outputPath/tensorflowmodeloptimize.log"
      val log = new java.io.File(logFilePath)
      timing("optimize tensorflow model") {
        Seq("sh",
          loadTensorflowModelScriptPath,
          frozenModelFilePath,
          pipelineConfigFilePath,
          extensionsConfigFilePath,
          outputPath) #> log !
      }
      logger.info(s"tensorflow model optimized, please check the output log $logFilePath")
      val modelFilePath: String = s"$outputPath/frozen_inference_graph.xml"
      val weightFilePath: String = s"$outputPath/frozen_inference_graph.bin"
      val mappingFilePath: String = s"$outputPath/frozen_inference_graph.mapping"
      val model = loadOpenVinoIR(modelFilePath, weightFilePath, deviceType)
      timing("delete temporary model files") {
        val modelFile = new File(modelFilePath)
        val weightFile = new File(weightFilePath)
        val mappingFile = new File(mappingFilePath)
        modelFile.delete()
        weightFile.delete()
        mappingFile.delete()
      }
      model
    }
  }

  def loadOpenVinoIR(modelFilePath: String,
                     weightFilePath: String,
                     deviceType: DeviceTypeEnumVal): OpenVinoInferenceModel = {
    timing("load openvino IR") {
      val supportive: OpenVinoInferenceSupportive = new OpenVinoInferenceSupportive()
      val executableNetworkReference: Long =
        supportive.loadOpenVinoIR(modelFilePath, weightFilePath, deviceType.value)
      new OpenVinoInferenceModel(executableNetworkReference, supportive)
    }
  }

  def load(path: String): Unit = {
    logger.info(s"start to load library: $path.")
    val inputStream = OpenVinoInferenceSupportive.getClass.getResourceAsStream(s"/${path}")
    val file = File.createTempFile("OpenVinoInferenceSupportiveLoader", path)
    val src = Channels.newChannel(inputStream)
    val dest = new FileOutputStream(file).getChannel
    dest.transferFrom(src, 0, Long.MaxValue)
    dest.close()
    src.close()
    val filePath = file.getAbsolutePath
    logger.info(s"loading library: $path from $filePath ...")
    try {
      System.load(filePath)
    } finally {
      file.delete()
    }
  }
}
