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

import java.io.{ByteArrayInputStream, File, FileOutputStream, IOException}
import java.nio.file.{Files, Paths}
import java.util.{ArrayList, Arrays, List => JList}

import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.zoo.pipeline.api.net.{NetUtils, SerializationHolder}
import com.intel.analytics.zoo.pipeline.inference.DeviceType.DeviceTypeEnumVal
import com.intel.analytics.zoo.pipeline.inference.OpenVINOModel.OpenVINOModelHolder
import com.intel.analytics.zoo.pipeline.inference.OpenVinoInferenceSupportive.logger
import org.apache.commons.io.FileUtils

import scala.collection.JavaConverters._
import scala.io.Source

class OpenVINOModel(var modelHolder: OpenVINOModelHolder,
                    var batchSize: Int = -1,
                    var isInt8: Boolean = false,
                    var deviceType: DeviceTypeEnumVal = DeviceType.CPU)
  extends AbstractModel with InferenceSupportive with Serializable {

  private var isRelease: Boolean = false

  @transient
  private lazy val supportive: OpenVinoInferenceSupportive = {
    println("Prepare OpenVINO bin " + this)
    new OpenVinoInferenceSupportive()
  }

  @transient
  private lazy val executableNetworkReference: Long = {
    println("OpenVINO loading in " + this)
    var nativeRef = -1L
    try {
      val modelFile = File.createTempFile("OpenVINO", "xml")
      Files.write(Paths.get(modelFile.toURI), modelHolder.modelBytes)
      val weightFile = File.createTempFile("OpenVINO", "bin")
      Files.write(Paths.get(modelFile.toURI), modelHolder.weightBytes)

      val buffer = Source.fromFile(modelFile)
      this.isInt8 = buffer.getLines().count(_ matches ".*statistics.*") > 0
      buffer.close()

      nativeRef = if (isInt8) {
        logger.info(s"Load int8 model")
        supportive.loadOpenVinoIRInt8(modelFile.getAbsolutePath,
          weightFile.getAbsolutePath,
          deviceType.value, batchSize)
      } else {
        supportive.loadOpenVinoIR(modelFile.getAbsolutePath,
          weightFile.getAbsolutePath,
          deviceType.value, batchSize)
      }
      FileUtils.deleteQuietly(modelFile)
      FileUtils.deleteQuietly(weightFile)
    }
    catch {
      case io: IOException =>
        System.out.println("error during loading OpenVINO model")
        throw io
    }
    nativeRef
  }

  override def predict(inputs: JList[JList[JTensor]]): JList[JList[JTensor]] = {
    val outputs = new ArrayList[JList[JTensor]]()
    inputs.asScala.map(input => {
      val tensor = input.get(0)
      val output = if (isInt8) {
        supportive.predictInt8(executableNetworkReference,
          tensor.getData, tensor.getShape)
      } else {
        supportive.predict(executableNetworkReference,
          tensor.getData, tensor.getShape)
      }
      outputs.add(Arrays.asList({
        output
      }))
    })
    outputs
  }

  override def predict(inputActivity: Activity): Activity = {
    val (inputList, batchSize) = inputActivity.isTable match {
      case true =>
        val inputTable = inputActivity.toTable
        val batchSize = inputTable.length()
        (transferBatchTableToJListOfJListOfJTensor(inputTable, batchSize), batchSize)
      case false =>
        val inputTensor = inputActivity.toTensor[Float]
        val batchSize = inputTensor.size(1)
        (transferBatchTensorToJListOfJListOfJTensor(inputTensor, batchSize), batchSize)
    }
    val outputs = predict(inputList)
    transferListOfActivityToActivityOfBatch(outputs, batchSize)
  }

  override def copy(num: Int): Array[AbstractModel] = Array(this)

  override def release(): Unit = {
    isReleased match {
      case true =>
      case false =>
        supportive.releaseOpenVINOIR(executableNetworkReference)
        isRelease = true
    }
  }

  override def isReleased(): Boolean = {
    isRelease
  }

  override def toString: String = s"OpenVinoInferenceModel with " +
    s"executableNetworkReference: $executableNetworkReference, supportive: $supportive"
}

object OpenVINOModel {

  @transient
  private lazy val inDriver = NetUtils.isDriver

  class OpenVINOModelHolder(@transient var modelBytes: Array[Byte],
                            @transient var weightBytes: Array[Byte])
    extends SerializationHolder {

    def getModelBytes(): Array[Byte] = {
      modelBytes
    }

    def getWeightBytes(): Array[Byte] = {
      weightBytes
    }

    override def writeInternal(out: CommonOutputStream): Unit = {
      if (inDriver) {
        out.writeInt(modelBytes.length)
        timing(s"writing ${modelBytes.length / 1024 / 1024}Mb openvino model to stream") {
          out.write(modelBytes)
        }
        out.writeInt(weightBytes.length)
        timing(s"writing ${weightBytes.length / 1024 / 1024}Mb openvino weight to stream") {
          out.write(weightBytes)
        }
      } else {
        out.writeInt(0)
      }
    }

    override def readInternal(in: CommonInputStream): Unit = {
      val modelLen = in.readInt()
      assert(modelLen >= 0, "OpenVINO model length should be an non-negative integer")
      modelBytes = new Array[Byte](modelLen)
      timing("reading OpenVINO model from stream") {
        var numOfBytes = 0
        while (numOfBytes < modelLen) {
          val read = in.read(modelBytes, numOfBytes, modelLen - numOfBytes)
          numOfBytes += read
        }
      }
      val weightLen = in.readInt()
      assert(weightLen >= 0, "OpenVINO weight length should be an non-negative integer")
      weightBytes = new Array[Byte](weightLen)
      timing("reading OpenVINO weight from stream") {
        var numOfBytes = 0
        while (numOfBytes < weightLen) {
          val read = in.read(weightBytes, numOfBytes, weightLen - numOfBytes)
          numOfBytes += read
        }
      }
    }
  }

  def apply(modelHolder: OpenVINOModelHolder): OpenVINOModel = {
    new OpenVINOModel(modelHolder)
  }

  def apply(modelBytes: Array[Byte], weightBytes: Array[Byte], batchSize: Int): OpenVINOModel = {
    OpenVinoInferenceSupportive.loadOpenVinoIR(modelBytes, weightBytes, DeviceType.CPU, batchSize)
  }

  def apply(modelHolder: OpenVINOModelHolder, batchSize: Int): OpenVINOModel = {
    OpenVinoInferenceSupportive.loadOpenVinoIR(modelHolder.getModelBytes(),
      modelHolder.getWeightBytes(),
      DeviceType.CPU, batchSize)
  }

  def apply(modelFilePath: String,
            weightFilePath: String,
            deviceType: DeviceTypeEnumVal,
            batchSize: Int = 0): OpenVINOModel = {
    OpenVinoInferenceSupportive.loadOpenVinoIR(modelFilePath,
      weightFilePath,
      deviceType,
      batchSize)
  }
}