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
import org.tensorflow.framework.GraphDef
import org.tensorflow.types.UInt8
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

  class ResourceManager() extends java.io.Serializable {
    private val tensorList: mutable.Set[TTensor[_]] = mutable.Set[TTensor[_]]()
    def createTFTensor(shape: Array[Long], buffer: FloatBuffer): TTensor[_] = {
      val TFTensor : TTensor[_] = TTensor.create(shape, buffer)
      tensorList += TFTensor
      return TFTensor
    }
    def createTFTensor(shape: Array[Long], buffer: ByteBuffer): TTensor[_] = {
      val TFTensor : TTensor[_] = TTensor.create(classOf[UInt8], shape, buffer)
      tensorList += TFTensor
      return TFTensor
    }
    def createTFTensor(shape: Array[Long], buffer: IntBuffer): TTensor[_] = {
      val TFTensor : TTensor[_] = TTensor.create(shape, buffer)
      tensorList += TFTensor
      return TFTensor
    }
    def createTFTensor(shape: Array[Long], buffer: LongBuffer): TTensor[_] = {
      val TFTensor : TTensor[_] = TTensor.create(shape, buffer)
      tensorList += TFTensor
      return TFTensor
    }
    def createTFTensor(shape: Array[Long], buffer: DoubleBuffer): TTensor[_] = {
      val TFTensor : TTensor[_] = TTensor.create(shape, buffer)
      tensorList += TFTensor
      return TFTensor
    }

    def createBoolTFTensor(shape: Array[Long], bytes: ByteBuffer): TTensor[_] = {
      val TFTensor : TTensor[_] = TTensor.create(classOf[java.lang.Boolean], shape, bytes)
      tensorList += TFTensor
      return TFTensor
    }

    def releaseTensor(t: TTensor[_]): Unit = {
      t.close()
      tensorList -= t
    }

    def isEmpty: Boolean = {
      tensorList.isEmpty
    }

    def destructTFTensors(): Unit = {
      for (tensor <- tensorList) {
        tensor.close()
      }

      tensorList.clear()
    }
  }


  @transient
  private lazy val tensorManager = new ResourceManager()

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

  def run(input: Vector[Tensor[Float]],
          inputTypes: Vector[DataType],
          output: Vector[Tensor[Float]],
          inputNames: Vector[String],
          outputNames: Vector[String],
          targets: Vector[String]): Vector[Tensor[Float]] = {
    NetUtils.timeIt("Graph Runner Run", GraphRunner.logger) {
      try {
        val runner = sess.runner()

        val inputTFTensors = new Array[TTensor[_]](inputNames.length)

        tensor2TFTensors(input, inputTypes, inputTFTensors)

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
          tf2bigdl(t.asInstanceOf[TTensor[Float]], output(idx))
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

  private def bigdl2Tf(t: Tensor[Float], dataType: DataType): TTensor[_] = {

    require(t.isContiguous(), "input to tfnet must be contiguous")
    val shape = t.size().map(_.toLong)
    val arr = t.storage().array()
    val offset: Int = t.storageOffset() - 1
    val length: Int = shape.product.toInt

    if (dataType == DataType.FLOAT) {
      val buffer = FloatBuffer.wrap(arr, offset, length)
      tensorManager.createTFTensor(shape, buffer)
    } else if (dataType == DataType.UINT8) {
      val buffer = ByteBuffer.wrap(GraphRunner.floatToUint8(arr), offset, length)
      tensorManager.createTFTensor(shape, buffer)
    } else if (dataType == DataType.INT32) {
      val buffer = IntBuffer.wrap(GraphRunner.floatToInt(arr), offset, length)
      tensorManager.createTFTensor(shape, buffer)
    } else if (dataType == DataType.INT64) {
      val buffer = LongBuffer.wrap(GraphRunner.floatToLong(arr), offset, length)
      tensorManager.createTFTensor(shape, buffer)
    } else if (dataType == DataType.DOUBLE) {
      val buffer = DoubleBuffer.wrap(GraphRunner.floatToDouble(arr), offset, length)
      tensorManager.createTFTensor(shape, buffer)
    } else if (dataType == DataType.BOOL) {
      val buffer = ByteBuffer.wrap(GraphRunner.floatToBool(arr), offset, length)
      tensorManager.createBoolTFTensor(shape, buffer)
    } else {
      throw new Exception(s"data type ${dataType} are not supported")
    }


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

  private def tensor2TFTensors(input: Seq[Tensor[Float]], types: Seq[DataType],
                                 tfTensors: Array[TTensor[_]]) = {
    val t = input
    require(tfTensors.length == t.length, "activity and tfTensors size does not equal," +
      s" activity length is ${t.length} tfTensors length is ${tfTensors.length}")
    var i = 0
    while (i < t.length) {
      val tfTensor = bigdl2Tf(t(i), types(i))
      if (tfTensors(i) != null) {
        tfTensors(i).close()
      }
      tfTensors(i) = tfTensor
      i += 1
    }
  }
}

object GraphRunner {

  assert(TFNetNative.isLoaded)

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
}
