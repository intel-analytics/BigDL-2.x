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

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.core.TFNetNative
import org.tensorflow.{DataType, Graph, Session, Tensor => TTensor}

import scala.collection.JavaConverters._
import org.json4s._
import org.slf4j.LoggerFactory

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

/**
 * [[TFNet]] wraps a tensorflow subgraph as a layer, and use tensorflow to
 * calculate the layer's output.
 *
 * This subgraph should not contain any tensorflow Variable and the input/output
 * must be numeric types
 *
 * When used with other layers for training, there should be no trainable layer
 * before this one, as the gradInput of this layer is always zero.
 *
 * @param graphDef serialized representation of a graph
 */
class GraphRunner(
            private val graphDef: Array[Byte],
            private val restoreOp: String,
            private val restorePathPlaceholder: String,
            private val saveOp: String,
            private val savePathPlaceholder: String,
            private val config: Array[Byte]) extends java.io.Serializable {

  @transient
  private lazy val tensorManager = new TFResourceManager()

  val output = ArrayBuffer[Tensor[Float]]()

  @transient
  private[zoo] lazy val sess = {
    val graph = new Graph()
    graph.importGraphDef(graphDef)
    val sess = new Session(graph, config)
    sess
  }

  def restoreFromFile(checkpointPath: String): Unit = {
    val runner = sess.runner()
    runner.addTarget(restoreOp)
    val pathTensor = org.tensorflow.Tensor.create(checkpointPath.getBytes())
    runner.feed(restorePathPlaceholder, pathTensor)
    runner.run()
    pathTensor.close()
  }

  def saveToFile(checkpointPath: String): Unit = {
    val runner = sess.runner()
    runner.addTarget(saveOp)
    val pathTensor = org.tensorflow.Tensor.create(checkpointPath.getBytes())
    runner.feed(savePathPlaceholder, pathTensor)
    runner.run()
    pathTensor.close()
  }

  def run(input: Vector[Tensor[_]],
          inputTypes: Vector[DataType],
          output: Vector[Tensor[Float]],
          inputNames: Vector[String],
          outputNames: Vector[String],
          targets: Vector[String]): Vector[Tensor[Float]] = {
    NetUtils.timeIt("Graph Runner Run", GraphRunner.logger) {
      try {
        val runner = sess.runner()

        val inputTFTensors = new Array[TTensor[_]](inputNames.length)

        tensorManager.tensor2TFTensors(input, inputTypes, inputTFTensors)

        // feed inputs
        inputNames.zipWithIndex.foreach { case (name, idx) =>
          runner.feed(name, inputTFTensors(idx))
        }

        // fetch outputs
        outputNames.foreach(runner.fetch)

        // add targets
        targets.foreach(runner.addTarget)


        val outputs = NetUtils.timeIt("Session Run", GraphRunner.logger) {
          runner.run()
        }

        outputs.asScala.zipWithIndex.foreach { case (t, idx) =>
          GraphRunner.tf2bigdl(t.asInstanceOf[TTensor[Float]], output(idx))
        }

        // outputs is returned by tensorflow and cannot be freed using tensorManager
        emptyTFTensorArray(outputs.asScala)

      } finally {
        tensorManager.destructTFTensors()
      }
    }
    output
  }

  private def emptyTFTensorArray(arr: mutable.Buffer[TTensor[_]]): Unit = {
    var i = 0
    while (i < arr.length) {
      tensorManager.releaseTensor(arr(i))
      arr(i) = null
      i += 1
    }
  }

  override def finalize(): Unit = {
    super.finalize()
    this.sess.close()
  }

  def release(): Unit = {
    this.sess.close()
  }
}

object GraphRunner {

  assert(TFNetNative.isLoaded)

  private[zoo] def tf2bigdl(t: TTensor[_], output: Tensor[Float]) = {
    val shape = t.shape().map(_.toInt)
    output.resize(shape)
    val buffer = FloatBuffer.wrap(
      output.storage().array(),
      output.storageOffset() - 1,
      shape.product)
    t.writeTo(buffer)
  }

  val logger = LoggerFactory.getLogger(getClass)

  implicit val formats = DefaultFormats

  val defaultSessionConfig = SessionConfig()

  case class SessionConfig(intraOpParallelismThreads: Int = 1,
                           interOpParallelismThreads: Int = 1,
                           usePerSessionThreads: Boolean = true) {

    // Ideally we should use the following code, however, importing tensorflow proto
    // will conflict with bigdl.

    //  val defaultSessionConfig = ConfigProto.newBuilder()
    //    .setInterOpParallelismThreads(1)
    //    .setIntraOpParallelismThreads(1)
    //    .setUsePerSessionThreads(true)
    //    .build().toByteArray

    def toByteArray(): Array[Byte] = {
      val intraSeq = if (intraOpParallelismThreads > 0) {
        Seq(16, intraOpParallelismThreads)
      } else {
        Seq[Int]()
      }
      val interSeq = if (interOpParallelismThreads > 0) {
        Seq(40, interOpParallelismThreads)
      } else {
        Seq[Int]()
      }
      val perSessSeq = if (usePerSessionThreads) {
        Seq(72, 1)
      } else {
        Seq[Int]()
      }

      (intraSeq ++ interSeq ++ perSessSeq).map(_.toByte).toArray
    }
  }
}
