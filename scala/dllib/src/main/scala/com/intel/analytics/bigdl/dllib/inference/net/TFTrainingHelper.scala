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

import com.intel.analytics.bigdl.dataset.{Sample, Transformer}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractCriterion, AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.T
import org.tensorflow.{Session, Tensor => TTensor}

import scala.collection.Iterator
import scala.collection.JavaConverters._

private[zoo] class TFTrainingHelper(tfnet: TFNet,
                                    inputs: Array[String],
                                    outputs: Array[String],
                                    variables: Array[String],
                                    gradVariables: Array[String])
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
                     variables: Array[String], gradVariables: Array[String])


