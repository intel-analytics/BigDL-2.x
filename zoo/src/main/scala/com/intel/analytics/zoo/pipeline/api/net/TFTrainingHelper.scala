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

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractCriterion, AbstractModule, Activity}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.ImageFeature
import com.intel.analytics.bigdl.utils.{T, Table}
import com.intel.analytics.zoo.feature.common.Preprocessing
import com.intel.analytics.zoo.feature.image.ImageProcessing
import com.intel.analytics.zoo.pipeline.api.keras.metrics.{Accuracy, BinaryAccuracy, CategoricalAccuracy, SparseCategoricalAccuracy}
import org.slf4j.LoggerFactory
import org.tensorflow.DataType

import scala.io.Source
import scala.reflect.io.Path

// variables and gradVariables need to be sorted by name if you want to use multiple
// optimization methods for a TensorFlow model according to variable names.
private[zoo] class TFTrainingHelper(graphRunner: GraphRunner,
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

  private val weightsMap = {
    val map = collection.mutable.Map[String, Tensor[Float]]()
    var i = 0
    while (i < variables.length) {
      map(variables(i)) = weights(i)
      i += 1
    }
    map
  }

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

  override def evaluate(): TFTrainingHelper.this.type = {
    super.evaluate()
    setVariableIntoTF(weights, variableAssignPlaceholders,
      variableTypes.map(NetUtils.tfenum2datatype), assignVariableOp)
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
      variableTypes.map(NetUtils.tfenum2datatype), assignVariableOp)
    setVariableIntoTF(extraParameters, extraVariableAssignPlaceholders,
      extraVariableTypes.map(NetUtils.tfenum2datatype), assignExtraVariableOP)
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

  override def apply(name: String): Option[AbstractModule[Activity, Activity, Float]] = {
    val targetVariables = if (name == getName()) variables else variables.filter(_.startsWith(name))
    if (targetVariables == null) {
      None
    }
    else {
      val targetWeights = targetVariables.map(weightsMap)
      Some(new TFSubGraph(targetWeights))
    }
  }

  override def updateOutput(input: Activity): Activity = {
    NetUtils.timeIt("updateOutput", TFTrainingHelper.logger) {
      if (this.isTraining()) {
        NetUtils.timeIt("setTrainingVariableIntoTF", TFTrainingHelper.logger) {
          setVariableIntoTF(weights, variableAssignPlaceholders,
            variableTypes.map(NetUtils.tfenum2datatype), assignVariableOp)
        }
      }

      if (!extraParameterRestored) {
        setVariableIntoTF(extraParameters, extraVariableAssignPlaceholders,
          extraVariableTypes.map(NetUtils.tfenum2datatype), assignExtraVariableOP)
        extraParameterRestored = true
      }

      val feeds = NetUtils.activity2VectorBuilder(input)

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

      val types = inputTypes.toVector.map(NetUtils.tfenum2datatype)

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
        NetUtils.timeIt("getExtraVariableFromTF", TFTrainingHelper.logger) {
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

object TFTrainingHelper {

  val logger = LoggerFactory.getLogger(getClass)

  def apply(modelPath: String, sessionConfig: Array[Byte] = null): TFTrainingHelper = {

    val folderPath = Path(modelPath)
    val trainingMetaPath = folderPath / Path("training_meta.json")
    val graphDefPath = folderPath / Path("model.meta")
    val checkpointPath = folderPath / Path("model")

    val jsonStr = Source.fromFile(trainingMetaPath.jfile).getLines().mkString
    import org.json4s._
    import org.json4s.jackson.JsonMethods._
    implicit val formats = DefaultFormats

    val trainingMeta = parse(jsonStr).camelizeKeys.extract[TrainMeta]

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

    val helper = new TFTrainingHelper(graphRunner,
      checkpointPath.toString(),
      trainingMeta.inputs,
      trainingMeta.inputTypes,
      trainingMeta.metricTensors ++ Array(trainingMeta.batchSizeTensor, trainingMeta.lossTensor),
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

case class TrainMeta(inputs: Array[String],
                     inputTypes: Array[Int],
                     metricTensors: Array[String],
                     batchSizeTensor: String,
                     lossTensor: String,
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


/**
 * TFSubGraph will only be used in DistriOptimizer for the purpose of training a TensorFlow
 * model using multiple optimization methods based on variable names.
 * Applying a TFTrainingHelper layer by name will get a corresponding instance of TFSubGraph.
 *
 * In DistriOptimizer.optimize(), TFSubGraph will only be used to get the sizes and offsets of
 * each weight portion, slice on the original weights and gradients and apply the optimization
 * method accordingly.
 * The gradients of TFSubGraph will never be used and thus a dummy Tensor is put as a placeholder.
 */
private class TFSubGraph(
    weights: Array[Tensor[Float]]) extends AbstractModule[Activity, Activity, Float] {
  override def updateOutput(input: Activity): Activity = {
    input
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    gradInput
  }

  override def parameters(): (Array[Tensor[Float]], Array[Tensor[Float]]) = {
    (weights, weights.map(_ => Tensor[Float]()))
  }
}


class IdentityCriterion extends AbstractCriterion[Activity, Activity, Float]() {

  override def updateOutput(input: Activity, target: Activity): Float = {
    if (input.isTensor) {
      input.toTensor[Float].value()
    } else {
      val table = input.toTable
      table[Tensor[Float]](table.length()).value()
    }
  }
  override def updateGradInput(input: Activity, target: Activity): Activity = {
    gradInput
  }
}

class TFValidationMethod(val valMethod: ValidationMethod[Float],
                         name: String,
                         outputIndices: java.util.List[Int],
                         labelIndices: java.util.List[Int]) extends ValidationMethod[Float] {

  private def toActivity(indices: java.util.List[Int], table: Table) = {
    if (indices.size() == 1) {
      table[Tensor[Float]](indices.get(0) + 1)
    } else {
      var i = 0
      val outputs = T()
      while (i < indices.size()) {
        outputs.insert(table(indices.get(i) + 1))
        i += 1
      }
      outputs
    }
  }

  private def oneBasedLabel(activity: Activity) = {
    if (activity.isTensor) {
      activity.toTensor[Float].add(1.0f)
    } else {
      val t = activity.toTable
      var i = 0
      while (i < t.length()) {
        t[Tensor[Float]](i + 1).add(1.0f)
        i += 1
      }
    }
  }

  override def apply(output: Activity, target: Activity): ValidationResult = {
    // the output layout [grads..., outputs..., labels..., loss]
    val outputT = output.toTable

    if (valMethod.isInstanceOf[Loss[Float]]) {
      val loss = outputT[Tensor[Float]](outputT.length()).value()
      return new LossResult(loss, 1)
    }

    val outputActivity: Activity = toActivity(outputIndices, outputT)
    val targetActivity: Activity = toActivity(labelIndices, outputT)

    val to1basedLabel = valMethod match {
      case _: SparseCategoricalAccuracy[Float] => false
      case _: CategoricalAccuracy[Float] => false
      case _: BinaryAccuracy[Float] => false
      case v: Accuracy[Float] => !v.zeroBasedLabel
      case _: Top1Accuracy[Float] => true
      case _: Top5Accuracy[Float] => true
      case _: TreeNNAccuracy[Float] => true
      case _ => false
    }

    if (to1basedLabel) {
      oneBasedLabel(targetActivity)
    }

    valMethod.apply(outputActivity, targetActivity)
  }

  override protected def format(): String = {
    (name + " " + valMethod.toString()).trim
  }
}

class StatelessMetric(name: String, idx: Int) extends ValidationMethod[Float] {
  override def apply(output: Activity, target: Activity): ValidationResult = {
    // the output layout [grads..., metrics]
    val outputT = output.toTable

    val value = outputT[Tensor[Float]](idx + 1).value()
    val count = outputT[Tensor[Float]](outputT.length() - 1).value().toInt

    new ContiguousResult(value * count, count, name)
  }

  override protected def format(): String = {
    name
  }
}

class MergeFeatureLabel() extends ImageProcessing {

  def createNewMergedSample(sample: Sample[Float]): Sample[Float] = {
    val newSize = sample.getFeatureSize() ++ sample.getLabelSize()
    Sample(sample.getData(), newSize, null)
  }

  override def transform(feature: ImageFeature): ImageFeature = {
    val oldSample = feature[Sample[Float]](ImageFeature.sample)
    val newSample = createNewMergedSample(oldSample)
    val newFeature = new ImageFeature()
    newFeature(ImageFeature.sample) = newSample
    newFeature
  }
}

class MergeFeatureLabelFeatureTransformer() extends Preprocessing[Any, Any] {

  private val mergeFun = new MergeFeatureLabel()
  override def apply(prev: Iterator[Any]): Iterator[Any] = {
    prev.map(transform)
  }

  private def transform(element: Any): Any = {
    element match {
      case feature: ImageFeature =>
        mergeFun.transform(feature)
      case sample: Sample[Float] =>
        mergeFun.createNewMergedSample(sample)
      case _ => throw new IllegalArgumentException(
        s"Element type ImageFeaute and Sample[Float] is supported. " +
          s"Element type ${element.getClass} is not supported.")
    }
  }
}
