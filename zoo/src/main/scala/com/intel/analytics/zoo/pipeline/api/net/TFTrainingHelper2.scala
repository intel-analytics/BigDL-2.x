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

package com.intel.analytics.zoo.pipeline.api.net

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import org.slf4j.LoggerFactory
import org.tensorflow.DataType

import scala.io.Source
import scala.reflect.io.Path

private[zoo] class TFTrainingHelper2(graphRunner: GraphRunner,
                                    checkpointPath: String,
                                    inputs: Array[String],
                                    inputTypes: Array[Int],
                                    outputs: Array[String],
                                    variables: Array[String],
                                    variableTypes: Array[Int],
                                    variableAssignPlaceholders: Array[String],
                                    assignVariableOp: String,
                                    extraVariables: Array[String],
                                    extraVariableTypes: Array[Int],
                                    extraVariableAssignPlaceholders: Array[String],
                                    assignExtraVariableOP: String,
                                    gradVariables: Array[String],
                                    updateOp: String,
                                    defaultTensorValue: Array[Array[Float]])
  extends AbstractModule[Activity, Activity, Float] {

  override def parameters(): (Array[Tensor[Float]], Array[Tensor[Float]]) = {
    (weights, gradWeights)
  }

  override def getExtraParameter(): Array[Tensor[Float]] = {
    extraParameters
  }

  private val extraParameters: Array[Tensor[Float]] = initVariables(extraVariables)

  private val weights = initVariables(variables)

  private def initVariables(variableNames: Array[String]): Array[Tensor[Float]] = {
    val ws = new Array[Tensor[Float]](variableNames.length)
    var i = 0
    while (i < ws.length) {
      ws(i) = Tensor[Float]()
      i += 1
    }
    ws
  }

  private val gradWeights = variables.map(_ => Tensor[Float]())

  private val graphOutputs = {
    val graphOuts = Vector.newBuilder[Tensor[Float]]

    var i = 0
    while (i < outputs.length) {
      graphOuts += Tensor[Float]()
      i += 1
    }

    i = 0
    while (i < gradVariables.length) {
      graphOuts += Tensor[Float]()
      i += 1
    }

    graphOuts.result()
  }

  private val gradWeightsBuffer =
    graphOutputs.slice(outputs.length, graphOutputs.length)

  output = {
    if (outputs.length == 1) {
      graphOutputs(0)
    } else {
      val out = T()
      var i = 0
      while (i < outputs.length) {
        out.insert(graphOutputs(i))
        i += 1
      }
      out
    }
  }

  override def evaluate(): TFTrainingHelper2.this.type = {
    super.evaluate()
    setVariableIntoTF(weights, variableAssignPlaceholders,
      variableTypes.map(TFTrainingHelper2.enum2datatype), assignVariableOp)
    this
  }


  private def getVariableFromTF(weights: Array[Tensor[Float]],
                                     variableNames: Array[String]) = {
    graphRunner.run(
      input = Vector.empty,
      inputTypes = Vector.empty,
      output = weights.toVector,
      inputNames = Vector.empty,
      outputNames = variableNames.toVector,
      targets = Vector.empty)
  }

  private def setVariableIntoTF(weights: Array[Tensor[Float]],
                                         inputNames: Array[String],
                                         variableTypes: Array[DataType],
                                         assignOp: String) = {
    graphRunner.run(
      input = weights.toVector,
      inputTypes = variableTypes.toVector,
      output = Vector.empty,
      inputNames = inputNames.toVector,
      outputNames = Vector.empty,
      targets = Vector(assignOp)
    )
  }

  def saveCheckpoint(): Unit = {
    setVariableIntoTF(weights, variableAssignPlaceholders,
      variableTypes.map(TFTrainingHelper2.enum2datatype), assignVariableOp)
    setVariableIntoTF(extraParameters, extraVariableAssignPlaceholders,
      extraVariableTypes.map(TFTrainingHelper2.enum2datatype), assignExtraVariableOP)
    graphRunner.saveToFile(checkpointPath)
  }

  @transient
  private var extraParameterRestored: Boolean = false

  def restoreFromCheckpoint(): Unit = {
    graphRunner.restoreFromFile(checkpointPath)
    if (weights.length > 0) {
      getVariableFromTF(weights, variableNames = variables)
    }

    if (extraParameters.length > 0) {
      getVariableFromTF(extraParameters, variableNames = extraVariables)
    }

    extraParameterRestored = true

  }

  override def updateOutput(input: Activity): Activity = {
    NetUtils.timeIt("updateOutput", TFTrainingHelper2.logger) {
      if (this.isTraining()) {
        NetUtils.timeIt("setTrainingVariableIntoTF", TFTrainingHelper2.logger) {
          setVariableIntoTF(weights, variableAssignPlaceholders,
            variableTypes.map(TFTrainingHelper2.enum2datatype), assignVariableOp)
        }
      }

      if (!extraParameterRestored) {
        setVariableIntoTF(extraParameters, extraVariableAssignPlaceholders,
          extraVariableTypes.map(TFTrainingHelper2.enum2datatype), assignExtraVariableOP)
        extraParameterRestored = true
      }

      val feeds = Vector.newBuilder[Tensor[Float]]
      if (input.isTensor) {
        feeds += input.toTensor[Float]
      } else {
        var i = 0
        while (i < input.toTable.length()) {
          feeds += input.toTable(i + 1)
          i += 1
        }

      }

      if (this.isTraining()) {
        var i = 0
        while (i < defaultTensorValue.length) {
          feeds += Tensor.scalar[Float](defaultTensorValue(i)(0))
          i += 1
        }
      } else {
        var i = 0
        while (i < defaultTensorValue.length) {
          feeds += Tensor.scalar[Float](defaultTensorValue(i)(1))
          i += 1
        }
      }

      val types = inputTypes.toVector.map(TFTrainingHelper2.enum2datatype)

      val (outputNames, outputTensors) = if (isTraining()) {
        (outputs.toVector ++ gradVariables.toVector, graphOutputs)
      } else {
        (outputs.toVector, graphOutputs.slice(0, outputs.length))
      }

      graphRunner.run(
        input = feeds.result(),
        inputTypes = types,
        output = outputTensors,
        inputNames = inputs.toVector,
        outputNames = outputNames,
        targets = Vector(updateOp))


      if (extraParameters.length > 0) {
        NetUtils.timeIt("getExtraVariableFromTF", TFTrainingHelper2.logger) {
          getVariableFromTF(extraParameters, variableNames = extraVariables)
        }
      }

      if (isTraining()) {
        gradWeights.zipWithIndex.foreach { case (grad, idx) =>
          grad.resizeAs(weights(idx)).add(gradWeightsBuffer(idx))
        }
      }
    }

    output
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    gradInput
  }
}

object TFTrainingHelper2 {

  val logger = LoggerFactory.getLogger(getClass)

  def enum2datatype(enum: Int): DataType = {
    enum match {
      case 1 => DataType.FLOAT
      case 2 => DataType.DOUBLE
      case 3 => DataType.INT32
      case 4 => DataType.UINT8
      case 7 => DataType.STRING
      case 9 => DataType.INT64
      case 10 => DataType.BOOL
      case _ => throw new IllegalArgumentException(s"unsupported tensorflow datatype $enum")

    }
  }

  def apply(modelPath: String, sessionConfig: Array[Byte] = null): TFTrainingHelper2 = {

    val folderPath = Path(modelPath)
    val trainingMetaPath = folderPath / Path("training_meta.json")
    val graphDefPath = folderPath / Path("model.meta")
    val checkpointPath = folderPath / Path("model")

    val jsonStr = Source.fromFile(trainingMetaPath.jfile).getLines().mkString
    import org.json4s._
    import org.json4s.jackson.JsonMethods._
    implicit val formats = DefaultFormats

    val trainingMeta = parse(jsonStr).camelizeKeys.extract[TrainMeta2]

    val graphDef = TFNet.parseGraph(graphDefPath.toString())
    val config = if (sessionConfig != null) {
      sessionConfig
    } else {
      TFNet.defaultSessionConfig.toByteArray()
    }

    val graphRunner = new GraphRunner(
      graphDef.toByteArray,
      trainingMeta.restoreOp,
      trainingMeta.restorePathPlaceholder,
      trainingMeta.saveOp,
      trainingMeta.savePathPlaceholder,
      config)

    val helper = new TFTrainingHelper2(graphRunner,
      checkpointPath.toString(),
      trainingMeta.inputs,
      trainingMeta.inputTypes,
      trainingMeta.outputs,
      trainingMeta.variables,
      trainingMeta.variableTypes,
      trainingMeta.variableAssignPlaceholders,
      trainingMeta.assignVariableOp,
      trainingMeta.extraVariables,
      trainingMeta.extraVariableTypes,
      trainingMeta.extraVariableAssignPlaceholders,
      trainingMeta.assignExtraVariableOp,
      trainingMeta.gradVariables,
      trainingMeta.updateOp,
      trainingMeta.defaultTensorValue
    )
    helper.restoreFromCheckpoint()
    helper
  }
}

case class TrainMeta2(inputs: Array[String],
                     inputTypes: Array[Int],
                     outputs: Array[String],
                     variables: Array[String],
                     variableTypes: Array[Int],
                     variableAssignPlaceholders: Array[String],
                     assignVariableOp: String,
                     extraVariables: Array[String],
                     extraVariableTypes: Array[Int],
                     extraVariableAssignPlaceholders: Array[String],
                     assignExtraVariableOp: String,
                     gradVariables: Array[String],
                     restoreOp: String,
                     restorePathPlaceholder: String,
                     saveOp: String,
                     savePathPlaceholder: String,
                     updateOp: String,
                     defaultTensorValue: Array[Array[Float]])


