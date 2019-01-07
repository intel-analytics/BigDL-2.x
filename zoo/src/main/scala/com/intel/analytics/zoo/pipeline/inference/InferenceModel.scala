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

import java.io.{IOException, ObjectInputStream, ObjectOutputStream}
import java.lang.{Float => JFloat, Integer => JInt}
import java.util
import java.util.concurrent.LinkedBlockingQueue
import java.util.{List => JList}

import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.zoo.pipeline.inference.DeviceType.DeviceTypeEnumVal

import scala.collection.JavaConverters._

class InferenceModel(private var supportedConcurrentNum: Int = 1,
                     private var originalModel: ExecutiveInferenceModel = null,
                     private[inference] var modelQueue:
                     LinkedBlockingQueue[ExecutiveInferenceModel] = null)
  extends InferenceSupportive with Serializable {
  this.modelQueue = new LinkedBlockingQueue[ExecutiveInferenceModel](supportedConcurrentNum)
  this.originalModel match {
    case null =>
    case _ => offerModelQueue()
  }

  def doLoad(modelPath: String, weightPath: String): Unit = {
    clearModelQueue()
    this.originalModel = InferenceModelFactory.loadFloatInferenceModel(modelPath, weightPath)
    offerModelQueue()
  }

  def doLoadCaffe(modelPath: String, weightPath: String): Unit = {
    clearModelQueue()
    this.originalModel = InferenceModelFactory.
      loadFloatInferenceModelForCaffe(modelPath, weightPath)
    offerModelQueue()
  }

  def doLoadTF(modelPath: String,
               intraOpParallelismThreads: Int,
               interOpParallelismThreads: Int,
               usePerSessionThreads: Boolean): Unit = {
    clearModelQueue()
    this.originalModel =
      InferenceModelFactory.loadFloatInferenceModelForTF(modelPath,
        intraOpParallelismThreads, interOpParallelismThreads, usePerSessionThreads)
    offerModelQueue()
  }

  def doLoadOpenvinoIR(modelFilePath: String,
                       weightFilePath: String,
                       deviceType: DeviceTypeEnumVal): Unit = {
    if (supportedConcurrentNum > 1) {
      InferenceSupportive.logger.warn(s"supportedConcurrentNum is $supportedConcurrentNum > 1, " +
        s"openvino model not supports shared weights model copies")
    }
    clearModelQueue()
    this.originalModel =
      InferenceModelFactory.loadOpenvinoInferenceModelForIR(
        modelFilePath, weightFilePath, deviceType)
    offerModelQueue()
  }

  def doLoadTensorflowModelAsOpenvino(frozenModelFilePath: String,
                                      pipelineConfigFilePath: String,
                                      extensionsConfigFilePath: String,
                                      deviceType: DeviceTypeEnumVal): Unit = {
    if (supportedConcurrentNum > 1) {
      InferenceSupportive.logger.warn(s"supportedConcurrentNum is $supportedConcurrentNum > 1, " +
        s"openvino model not supports shared weights model copies")
    }
    clearModelQueue()
    this.originalModel = InferenceModelFactory.loadOpenvinoInferenceModelForTF(
      frozenModelFilePath, pipelineConfigFilePath, extensionsConfigFilePath, deviceType)
    offerModelQueue()
  }

  def doReload(modelPath: String, weightPath: String): Unit = {
    clearModelQueue()
    doLoad(modelPath, weightPath)
  }

  @deprecated
  def doPredict(input: JList[JFloat], shape: JList[JInt]): JList[JFloat] = {
    timing("model predict") {
      val inputTensor = new JTensor(input, shape)
      val inputList = util.Arrays.asList({
        inputTensor
      })
      val inputs = util.Arrays.asList({
        inputList
      })
      val results = predict(inputs)
      results.get(0).get(0).getData.toList.asJava.asInstanceOf[JList[JFloat]]
    }
  }

  def doPredict(inputs: JList[JList[JTensor]]): JList[JList[JTensor]] = {
    timing(s"model predict for batch ${inputs.size()}") {
      val batchSize = inputs.size()
      require(batchSize > 0, "inputs size should > 0")
      predict(inputs)
    }
  }

  def doPredict(inputActivity: Activity): Activity = {
    var model: ExecutiveInferenceModel = null
    try {
      model = modelQueue.take
    } catch {
      case e: InterruptedException => throw new InferenceRuntimeException("no model available", e);
    }
    try {
      val result = model.predict(inputActivity)
      result
    } finally {
      modelQueue.offer(model)
    }
  }

  private def predict(inputs: JList[JList[JTensor]]): JList[JList[JTensor]] = {
    var model: ExecutiveInferenceModel = null
    try {
      model = modelQueue.take
    } catch {
      case e: InterruptedException => throw new InferenceRuntimeException("no model available", e);
    }
    try {
      model.predict(inputs)
    } finally {
      modelQueue.offer(model)
    }
  }

  private def clearModelQueue(): Unit = {
    this.originalModel match {
      case null =>
      case _ => this.originalModel.release(); this.originalModel = null
    }
    List.range(0, this.modelQueue.size()).map(i => {
      val model = this.modelQueue.take
      this.modelQueue.remove(model)
      model.release()
    })
    this.modelQueue.clear()
  }

  private def offerModelQueue(): Unit = {
    require(this.originalModel != null, "original model can not be null")
    require(this.supportedConcurrentNum > 0, "supported concurrent number should > 0")
    val models = this.originalModel.copy(supportedConcurrentNum)
    models.map(this.modelQueue.offer(_))
  }

  @throws(classOf[IOException])
  private def writeObject(out: ObjectOutputStream): Unit = {
    out.writeInt(supportedConcurrentNum)
    out.writeObject(originalModel)
  }

  @throws(classOf[IOException])
  private def readObject(in: ObjectInputStream): Unit = {
    System.setProperty("bigdl.localMode", System.getProperty("bigdl.localMode", "true"))
    System.setProperty("bigdl.coreNumber", System.getProperty("bigdl.coreNumber", "1"))
    Engine.init
    this.supportedConcurrentNum = in.readInt
    this.originalModel = in.readObject.asInstanceOf[FloatInferenceModel]
    this.modelQueue = new LinkedBlockingQueue[ExecutiveInferenceModel](supportedConcurrentNum)
    offerModelQueue()
  }

  override def toString: String =
    s"InferenceModel($supportedConcurrentNum, $originalModel, $modelQueue)"

}
