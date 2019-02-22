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

import java.nio._

import com.intel.analytics.bigdl.dataset.{MiniBatch, Sample, Transformer}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractCriterion, AbstractModule, Activity}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.zoo.pipeline.api.keras.metrics.Accuracy
import com.intel.analytics.zoo.pipeline.api.net.TFNet.ClosableGraph
import org.apache.spark.rdd.RDD
import org.tensorflow.types.UInt8
import org.tensorflow.{DataType, Session, Tensor => TTensor}

import scala.collection.{Iterator, mutable}
import scala.collection.JavaConverters._
import scala.io.Source
import scala.reflect.io.Path

private[zoo] class TFTrainingHelper(tfnet: TFNet,
                                    inputs: Array[String],
                                    outputs: Array[String],
                                    variables: Array[String],
                                    gradVariables: Array[String],
                                    defaultTensorValue: Array[Array[Float]])
  extends AbstractModule[Activity, Activity, Float] {

  override def parameters(): (Array[Tensor[Float]], Array[Tensor[Float]]) = {
    (weights, gradWeights)
  }
  private val weights = {
    val ws = new Array[Tensor[Float]](variables.length)
    var i = 0
    while (i < ws.length) {
      ws(i) = Tensor[Float]()
      i += 1
    }
    setWeights(ws)
  }

  private val gradWeights = variables.map(_ => Tensor[Float]())


  private def setWeights(weights: Array[Tensor[Float]]) = {
    val sess = tfnet.sess
    val runner = sess.runner()
    variables.foreach(runner.fetch)
    runner.run().asScala.zipWithIndex.map { case (fetch, idx) =>
      val t = weights(idx)
      tf2bigdl(fetch.asInstanceOf[TTensor[Float]], t)
      t
    }
    weights
  }

  private def tf2bigdl(t: TTensor[_], output: Tensor[Float]) = {
    val shape = t.shape().map(_.toInt)
    output.resize(shape)
    val buffer = FloatBuffer.wrap(
      output.storage().array(),
      output.storageOffset() - 1,
      shape.product)
    t.writeTo(buffer)
  }

  override def updateOutput(input: Activity): Activity = {
    val feeds = T()
    if (input.isTensor) {
      feeds.insert(input)
    } else {
      var i = 0
      while (i < input.toTable.length()) {
        feeds.insert(input.toTable(i + 1))
        i += 1
      }

    }

    if (this.isTraining()) {
      var i = 0
      while (i < defaultTensorValue.length) {
        feeds.insert(Tensor.scalar[Float](defaultTensorValue(i)(0)))
        i += 1
      }
    } else {
      var i = 0
      while (i < defaultTensorValue.length) {
        feeds.insert(Tensor.scalar[Float](defaultTensorValue(i)(1)))
        i += 1
      }
    }

    var i = 0
    while (i < weights.length) {
      feeds.insert(weights(i))
      i += 1
    }

    val fetches = tfnet.forward(feeds).toTable.toSeq[Tensor[Float]].toArray

    gradWeights.zipWithIndex.foreach { case (grad, idx) =>
      grad.resizeAs(weights(idx)).add(fetches(idx))
    }

    val realOutputs = fetches.slice(weights.length, fetches.length)

    output = if (realOutputs.length == 1) {
      realOutputs.head
    } else {
      val result = T()
      var i = 0
      while (i < realOutputs.length) {
        result.insert(realOutputs(i))
        i += 1
      }
      result
    }
    output
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    gradInput
  }
}

object TFTrainingHelper {

  def apply(modelPath: String, sessionConfig: Array[Byte] = null): TFTrainingHelper = {
    val (model, meta) = NetUtils.processTFFolder(modelPath)

    val folderPath = Path(modelPath)
    val trainingMetaPath = folderPath / Path("training_meta.json")

    val jsonStr = Source.fromFile(trainingMetaPath.jfile).getLines().mkString
    import org.json4s._
    import org.json4s.jackson.JsonMethods._
    implicit val formats = DefaultFormats

    val trainingMeta = parse(jsonStr).camelizeKeys.extract[TrainMeta]

    val newMeta = Meta(
      (meta.inputNames.toSeq ++:
        trainingMeta.variables.toSeq).toArray,
      meta.outputNames)
    val graphDef = TFNet.parseGraph(model)
    val config = if (sessionConfig != null) {
      sessionConfig
    } else {
      TFNet.defaultSessionConfig.toByteArray()
    }
    val tfnet = TFNet(graphDef, model, newMeta, config)


    new TFTrainingHelper(tfnet,
      trainingMeta.inputNames,
      trainingMeta.outputNames,
      trainingMeta.variables,
      trainingMeta.gradVariables,
      trainingMeta.defaultTensorValues
    )
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
                         outputLength: Int,
                         targetLength: Int) extends ValidationMethod[Float] {
  override def apply(output: Activity, target: Activity): ValidationResult = {
    // the output layout [grads..., outputs..., labels..., loss]
    val outputT = output.toTable

    if (valMethod.isInstanceOf[Loss[Float]]) {
      val loss = outputT[Tensor[Float]](outputT.length()).value()
      return new LossResult(loss, 1)
    }
    val outputActivity: Activity = if (outputLength == 1) {
      outputT[Tensor[Float]](outputT.length() - outputLength - targetLength)
    } else {
      var i = outputT.length() - outputLength - targetLength
      val outputs = T()
      while (i < outputT.length() - targetLength) {
        outputs.insert(outputT(i))
          i += 1
      }
      outputs
    }

    val to1basedLabel = !valMethod.isInstanceOf[Accuracy[Float]] &&
      valMethod.isInstanceOf[Top1Accuracy[Float]] ||
        valMethod.isInstanceOf[Top5Accuracy[Float]] ||
        valMethod.isInstanceOf[TreeNNAccuracy[Float]]
    val targetActivity = if (targetLength == 1) {
      val t = outputT[Tensor[Float]](outputT.length() - targetLength)
      if (to1basedLabel) t.add(1.0f)
      t
    } else {
      var i = outputT.length() - targetLength
      val targets = T()
      while (i < outputT.length()) {
        val t = outputT[Tensor[Float]](i)
        if (to1basedLabel) t.add(1.0f)
        targets.insert(t)
        i += 1
      }
      targets
    }

    valMethod.apply(outputActivity, targetActivity)
  }

  override protected def format(): String = {
    valMethod.toString()
  }
}

class MergeFeatureLabel() extends Transformer[Sample[Float], Sample[Float]] {
  override def apply(prev: Iterator[Sample[Float]]): Iterator[Sample[Float]] = {
    new Iterator[Sample[Float]] {

      override def hasNext: Boolean = prev.hasNext

      override def next(): Sample[Float] = {
        val oldSample = prev.next()
        val newSize = oldSample.getFeatureSize() ++ oldSample.getLabelSize()
        Sample(oldSample.getData(), newSize, null)
      }
    }
  }
}

case class TrainMeta(inputNames: Array[String], outputNames: Array[String],
                     variables: Array[String], gradVariables: Array[String],
                     defaultTensorValues: Array[Array[Float]])


class TFOptimizer(modelPath: String,
                  optimMethod: OptimMethod[Float],
                  x: RDD[Sample[Float]],
                  batchSize: Int = 32) {
  private val trainer: TFTrainingHelper = TFTrainingHelper(modelPath)
  private val optimizer: Optimizer[Float, MiniBatch[Float]] = {
    val optimizer = Optimizer[Float](trainer, x, new IdentityCriterion(), batchSize)

    optimizer.setOptimMethod(optimMethod)
    optimizer
  }

  def optimize(endTrigger: Trigger = Trigger.maxEpoch(1)): Array[Tensor[Float]] = {
    optimizer.setEndWhen(endTrigger)
    optimizer.optimize()
    trainer.parameters()._1
  }
}


class TFStatefulValidationMethod(val metricName: String,
                                 @ transient var graph: ClosableGraph,
                                 val outputNames: Array[String],
                                 val targetNames: Array[String],
                                 val outputIndices: Array[Int],
                                 val targetIndices: Array[Int],
                                 val metricsTensorNames: Array[String],
                                 val initVariables: Map[String, (Tensor[Float], DataType)],
                                 val assignVariableOp: String,
                                 val allOutputLength: Int,
                                 val allTargetLength: Int
                         ) extends ValidationMethod[Float] {


  override def apply(output: Activity, target: Activity): ValidationResult = {
    // the output layout [grads..., outputs..., labels..., loss]

    val outputT = output.toTable

    val outputActivity = if (allOutputLength == 1) {
      outputT[Tensor[Float]](outputT.length() - allOutputLength - allTargetLength)
    } else {
      val offset = outputT.length() - allOutputLength - allTargetLength
      val outputs = T()
      var i = 0
      while (i < outputIndices.length) {
        outputs.insert(outputT(outputIndices(i)))
        i += 1
      }
      outputs
    }

    val targetActivity = if (allTargetLength == 1) {
      val t = outputT[Tensor[Float]](outputT.length() - allTargetLength)
      t
    } else {
      val offset = outputT.length() - allTargetLength
      val targets = T()
      var i = 0
      while (i < targetIndices.length) {
        val t = outputT[Tensor[Float]](targetIndices(i))
        targets.insert(t)
        i += 1
      }
      targets
    }

    null
  }

  override protected def format(): String = {
    metricName
  }

}

class TFStatefulValidationResult(@transient var graph: ClosableGraph,
                                 val outputNames: Array[String],
                                 val targetNames: Array[String],
                                 val updateOp: String,
                                 val metricsTensorName: String,
                                 var outputs: mutable.ArrayBuffer[Array[Tensor[Float]]],
                                 var targets: mutable.ArrayBuffer[Array[Tensor[Float]]],
                                 val initVariables: Map[String, (Tensor[Float], Int)],
                                 val assignVariableOp: String,
                                 val metricName: String
                        ) extends ValidationResult {

  private var currentResult: (Float, Int) = (0.0f, -1)

  private def resetState(sess: Session) = {
    val runner = sess.runner()
    val variableTensors = initVariables.mapValues(valueAndType =>
      Utils.bigdl2Tf(valueAndType._1, new DataType(valueAndType._2)))

    for ((key, value) <- variableTensors) {
      runner.feed(key, value)
    }

    runner.fetch(assignVariableOp)

    variableTensors.foreach(x => x._2.close())
  }

  private def updateState(sess: Session) = {
    var i = 0
    while (i < outputs.length) {
      val runner = sess.runner()
      outputs.apply(i)
      val outputTFTensors = outputs(i).map(Utils.bigdl2Tf(_, DataType.FLOAT))
      val targetTFTensors = targets(i).map(Utils.bigdl2Tf(_, DataType.FLOAT))

      var j = 0
      while (j < outputNames.length) {
        runner.feed(outputNames(j), outputTFTensors(j))
        j += 1
      }

      j = 0
      while (j < targetNames.length) {
        runner.feed(targetNames(j), targetTFTensors(j))
        j += 1
      }

      runner.fetch(updateOp)

      runner.run()

      outputTFTensors.foreach(_.close())
      targetTFTensors.foreach(_.close())

      i += 1
    }
  }

  private def getResult(sess: Session): Float = {
    val runner = sess.runner()
    runner.fetch(metricsTensorName)
    val tfTensor = runner.run().get(0)
    val tensor = Tensor[Float]()
    Utils.tf2bigdl(tfTensor, tensor)
    tfTensor.close()
    tensor.value()
  }

  private def calcCurrentResult() = {

    val sess = new Session(graph.graph)
    resetState(sess)
    updateState(sess)
    val value = getResult(sess)
    val count = outputs.flatMap(_.map(_.size(1))).sum
    (value, count)
  }

  override def result(): (Float, Int) = {
    if (currentResult._2 >= 0) {
      currentResult
    } else {
      calcCurrentResult()
      currentResult
    }
  }

  // scalastyle:off methodName
  override def +(other: ValidationResult): ValidationResult = {
    val tfResult = other.asInstanceOf[TFStatefulValidationResult]
    this.outputs ++= tfResult.outputs
    this.targets ++= tfResult.targets
    currentResult = (0.0f, -1)
    this
  }
  // scalastyle:on methodName

  override protected def format(): String = {
    val (value, count) = if (currentResult._2 >= 0) {
      currentResult
    } else {
      calcCurrentResult()
      currentResult
    }
    s"Metrics(Name: $metricName, Value: $value, Count: $count)"
  }
}

object Utils {

  private def createTFTensor(shape: Array[Long], buffer: FloatBuffer): TTensor[_] = {
    val TFTensor : TTensor[_] = TTensor.create(shape, buffer)
    TFTensor
  }

  private def createTFTensor(shape: Array[Long], buffer: ByteBuffer): TTensor[_] = {
    val TFTensor : TTensor[_] = TTensor.create(classOf[UInt8], shape, buffer)
    TFTensor
  }

  private def createTFTensor(shape: Array[Long], buffer: IntBuffer): TTensor[_] = {
    val TFTensor : TTensor[_] = TTensor.create(shape, buffer)
    TFTensor
  }

  private def createTFTensor(shape: Array[Long], buffer: LongBuffer): TTensor[_] = {
    val TFTensor : TTensor[_] = TTensor.create(shape, buffer)
    TFTensor
  }

  private def createTFTensor(shape: Array[Long], buffer: DoubleBuffer): TTensor[_] = {
    val TFTensor : TTensor[_] = TTensor.create(shape, buffer)
    TFTensor
  }

  private def createBoolTFTensor(shape: Array[Long], bytes: ByteBuffer): TTensor[_] = {
    val TFTensor : TTensor[_] = TTensor.create(classOf[java.lang.Boolean], shape, bytes)
    TFTensor
  }

  private def floatToInt(array: Array[Float]): Array[Int] = {
    val result = new Array[Int](array.length)
    var i = 0
    while (i < array.length) {
      result(i) = array(i).toInt
      i = i + 1
    }
    result
  }

  private def floatToLong(array: Array[Float]): Array[Long] = {
    val result = new Array[Long](array.length)
    var i = 0
    while (i < array.length) {
      result(i) = array(i).toLong
      i = i + 1
    }
    result
  }

  private def floatToDouble(array: Array[Float]): Array[Double] = {
    val result = new Array[Double](array.length)
    var i = 0
    while (i < array.length) {
      result(i) = array(i).toDouble
      i = i + 1
    }
    result
  }

  private def floatToUint8(array: Array[Float]): Array[Byte] = {
    val result = new Array[Byte](array.length)
    var i = 0
    while (i < array.length) {
      result(i) = array(i).toByte
      i = i + 1
    }
    result
  }

  private def floatToBool(array: Array[Float]): Array[Byte] = {
    val result = new Array[Byte](array.length)
    var i = 0
    while (i < array.length) {
      result(i) = if (array(i) == 0.0) 0.toByte else 1.toByte
      i = i + 1
    }
    result
  }


  def bigdl2Tf(t: Tensor[Float], dataType: DataType): TTensor[_] = {

    require(t.isContiguous(), "input to tfnet must be contiguous")
    val shape = t.size().map(_.toLong)
    val arr = t.storage().array()
    val offset: Int = t.storageOffset() - 1
    val length: Int = shape.product.toInt

    if (dataType == DataType.FLOAT) {
      val buffer = FloatBuffer.wrap(arr, offset, length)
      createTFTensor(shape, buffer)
    } else if (dataType == DataType.UINT8) {
      val buffer = ByteBuffer.wrap(floatToUint8(arr), offset, length)
      createTFTensor(shape, buffer)
    } else if (dataType == DataType.INT32) {
      val buffer = IntBuffer.wrap(floatToInt(arr), offset, length)
      createTFTensor(shape, buffer)
    } else if (dataType == DataType.INT64) {
      val buffer = LongBuffer.wrap(floatToLong(arr), offset, length)
      createTFTensor(shape, buffer)
    } else if (dataType == DataType.DOUBLE) {
      val buffer = DoubleBuffer.wrap(floatToDouble(arr), offset, length)
      createTFTensor(shape, buffer)
    } else if (dataType == DataType.BOOL) {
      val buffer = ByteBuffer.wrap(floatToBool(arr), offset, length)
      createBoolTFTensor(shape, buffer)
    } else {
      throw new Exception(s"data type ${dataType} are not supported")
    }
  }

  def tf2bigdl(t: TTensor[_], output: Tensor[Float]): Unit = {
    val shape = t.shape().map(_.toInt)
    output.resize(shape)
    val buffer = FloatBuffer.wrap(
      output.storage().array(),
      output.storageOffset() - 1,
      shape.product)
    t.writeTo(buffer)
  }
}


