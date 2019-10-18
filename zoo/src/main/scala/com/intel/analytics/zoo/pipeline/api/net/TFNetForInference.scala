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
import org.tensorflow.framework.GraphDef
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Placeholder
import org.tensorflow.{DataType, SavedModelBundle}


private[zoo] class TFNetForInference(graphRunner: GraphRunner,
                                    inputs: Array[String],
                                    inputTypes: Array[Int],
                                    outputs: Array[String],
                                    variables: Array[String],
                                    variableTypes: Array[Int],
                                    variableAssignPlaceholders: Array[String],
                                    assignVariableOps: Array[String],
                                    private val weights: Array[Tensor[Float]])
  extends AbstractModule[Activity, Activity, Float] {

  override def parameters(): (Array[Tensor[Float]], Array[Tensor[Float]]) = {
    (weights, gradWeights)
  }

  private val gradWeights = variables.map(_ => Tensor[Float]())

  private val graphOutputs = {
    val graphOuts = Vector.newBuilder[Tensor[Float]]

    var i = 0
    while (i < outputs.length) {
      graphOuts += Tensor[Float]()
      i += 1
    }

    graphOuts.result()
  }

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

  private def setVariableIntoTF(weights: Array[Tensor[Float]],
                                inputNames: Array[String],
                                variableTypes: Array[DataType],
                                assignOps: Array[String]) = {
    graphRunner.run(
      input = weights.toVector,
      inputTypes = variableTypes.toVector,
      output = Vector.empty,
      inputNames = inputNames.toVector,
      outputNames = Vector.empty,
      targets = assignOps.toVector
    )
  }

  setVariableIntoTF(weights, variableAssignPlaceholders,
    variableTypes.map(NetUtils.tfenum2datatype), assignVariableOps)

  override def updateOutput(input: Activity): Activity = {
    NetUtils.timeIt("updateOutput", TFNetForInference.logger) {

      val feeds = NetUtils.activity2VectorBuilder(input)

      val types = inputTypes.toVector.map(NetUtils.tfenum2datatype)

      graphRunner.run(
        input = feeds.result(),
        inputTypes = types,
        output = graphOutputs,
        inputNames = inputs.toVector,
        outputNames = outputs.toVector,
        targets = Vector.empty)
    }

    output
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    gradInput
  }
}

object TFNetForInference {

  TFNet

  val logger = LoggerFactory.getLogger(getClass)

  import scala.collection.JavaConverters._

  val frameworkDataType2Class = Map(
    org.tensorflow.framework.DataType.DT_FLOAT -> classOf[java.lang.Float],
    org.tensorflow.framework.DataType.DT_INT32 -> classOf[java.lang.Integer],
    org.tensorflow.framework.DataType.DT_INT64 -> classOf[java.lang.Long]
  )

  val frameworkDataType2DataType = Map(
    org.tensorflow.framework.DataType.DT_FLOAT -> org.tensorflow.DataType.FLOAT,
    org.tensorflow.framework.DataType.DT_INT32 -> org.tensorflow.DataType.INT32,
    org.tensorflow.framework.DataType.DT_INT64 -> org.tensorflow.DataType.INT64
  )

  def fromSavedModel(modelPath: String, tag: String,
                     inputs: Array[String],
                     outputs: Array[String],
                     sessionConfig: Array[Byte]): TFNetForInference = {

    val savedModelBundle = SavedModelBundle.load(modelPath, tag)

    val graph = savedModelBundle.graph()
    val ops = Ops.create(graph).withSubScope("analytics-zoo")

    val variableTypes = Set("Variable", "VariableV2", "VarHandleOp")
    val graphBytes = graph.toGraphDef

    val graphDef = GraphDef.parseFrom(graphBytes)

    val newOps = graphDef.getNodeList.asScala.filter{ node =>
      variableTypes(node.getOp)
    }.map{ x =>
      val name = x.getName
      val dataType = x.getAttrMap.get("dtype").getType
      val opType = x.getOp
      val operation = graph.operation(name)
      val dataTypeClass = frameworkDataType2Class(dataType)
      val operationOutput = operation.output(0)
      if (opType == "VarHandleOp") {
        val readVariable = ops.readVariableOp(operationOutput, dataTypeClass)
        val floatVariable = ops.cast(readVariable, classOf[java.lang.Float])
        val placeholder = ops.placeholder(dataTypeClass,
          Placeholder.shape(readVariable.asOutput().shape()))

        // do it manually to get a reference of the op and get the op name
        val builder = ops.scope().graph().opBuilder("AssignVariableOp",
          ops.scope().makeOpName("AssignVariableOp"))
        builder.addInput(operationOutput)
        builder.addInput(placeholder.asOutput())
        val assignOp = builder.build()
        (floatVariable.asOutput().op().name(),
          placeholder.asOutput().op().name(), assignOp.name(),
          dataType, operationOutput.shape(), operation.name())
      } else {
        val readVariable = operationOutput
        val floatVariable = ops.cast(readVariable, classOf[java.lang.Float])
        val placeholder = ops.placeholder(dataTypeClass, Placeholder.shape(operationOutput.shape()))

        // do it manually to get a reference of the op and get the op name
        val builder = ops.scope().graph().opBuilder("Assign",
          ops.scope().makeOpName("Assign"))
        builder.addInput(operationOutput)
        builder.addInput(placeholder.asOutput())
        val assignOp = builder.build()
        (floatVariable.asOutput().op().name(),
          placeholder.asOutput().op().name(), assignOp.name(),
          dataType, operationOutput.shape(), operation.name())
      }
    }

    val readVariableNames = newOps.map(_._1)
    val placeholderNames = newOps.map(_._2)
    val assign = newOps.map(_._3)
    val dataTypes = newOps.map(_._4)
    val dataShapes = newOps.map(x => (x._5, x._6))

    val graphdef = GraphDef.parseFrom(ops.scope().graph().toGraphDef)

    val graphRunner = new GraphRunner(ops.scope().graph().toGraphDef, null, null, null, null,
      TFNet.defaultSessionConfig.toByteArray())

    val session = savedModelBundle.session()

    val weights = readVariableNames.zip(dataShapes).map { case (name, (shape, orignalName)) =>
      val runner = session.runner()
      runner.fetch(name)
      try {
        val value = runner.run()
        val bigdlTensor = Tensor[Float]()
        GraphRunner.tf2bigdl(value.get(0), bigdlTensor)
        value.get(0).close()
        bigdlTensor
      } catch {
        case _: Exception =>
          TFNetForInference.logger.warn(s"Cannot find variable value for <$orignalName>, " +
            s"using default value zero")
          val shapeArr = new Array[Int](shape.numDimensions())
          var i = 0
          while (i < shape.numDimensions()) {
            shapeArr(i) = shape.size(i).toInt
            i += 1
          }
          Tensor[Float](sizes = shapeArr)
      }
    }.toArray

    // get weights out
    graphRunner.run(
      input = weights.toVector,
      inputTypes = dataTypes.map(frameworkDataType2DataType).toVector,
      output = Vector.empty,
      inputNames = placeholderNames.toVector,
      outputNames = Vector.empty,
      targets = assign.toVector)

    val inputTypes = inputs.map { name =>
      val opAndPort = name.split(":")
      val op = opAndPort.head
      val port = opAndPort(1)
      val opRef = graph.operation(op)
      if (opRef == null) {
        throw new IllegalArgumentException(s"Cannot find input op <$name>")
      }
      NetUtils.tfdatatype2enum(opRef.output(port.toInt).dataType())
    }

    // clean up native resources
    savedModelBundle.close()

    new TFNetForInference(graphRunner = graphRunner,
      inputs = inputs,
      inputTypes = inputTypes,
      outputs = outputs,
      variables = readVariableNames.toArray,
      variableTypes = dataTypes.map(_.getNumber).toArray,
      variableAssignPlaceholders = placeholderNames.toArray,
      assignVariableOps = assign.toArray,
      weights = weights)
  }
}


