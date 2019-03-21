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

import java.nio.FloatBuffer

import com.intel.analytics.bigdl.dataset.{MiniBatch, Sample, Transformer}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractCriterion, AbstractModule, Activity}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.python.api.{PythonBigDLKeras, Sample => JSample}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.zoo.pipeline.api.keras.metrics.{Accuracy, BinaryAccuracy, CategoricalAccuracy, SparseCategoricalAccuracy}
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.rdd.RDD
import org.tensorflow.{Session, Tensor => TTensor}

import scala.collection.Iterator
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
      while (i < outputLength - targetLength) {
        outputs.insert(outputT(i))
          i += 1
      }
      outputs
    }

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

    val targetActivity = if (targetLength == 1) {
      val t = outputT[Tensor[Float]](outputT.length() - targetLength)
      if (to1basedLabel) t.add(1.0f)
      t
    } else {
      var i = outputT.length() - targetLength
      val targets = T()
      while (i < outputLength) {
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


