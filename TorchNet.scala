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

import java.io._
import java.nio.channels.Channels
import java.nio.file.{Files, Paths}

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.pipeline.api.Predictable
import com.intel.analytics.zoo.pipeline.api.net.TorchNet.TorchModelHolder
import org.apache.commons.io.FileUtils

import scala.reflect.ClassTag

/**
 * [[TorchNet]] wraps a TorchScript model as a single layer.
 */
class TorchNet private(private val modelHolder: TorchModelHolder)
    extends AbstractModule[Tensor[Float], Tensor[Float], Float] with Predictable[Float] {

  protected val module: Module[Float] = this
  implicit val ev = TensorNumeric.NumericFloat
  implicit val tag: ClassTag[Float] = ClassTag.Float

  var weights: Tensor[Float] = _
  var gradients: Tensor[Float] = _

  /**
   * sequential id in cpp: std::vector<std::shared_ptr<torch::jit::script::Module>> handles;
   * mark the model as transient and reload TorchNet from byteArray on executors
   */
  @transient
  lazy val nativeRef: Long = {
    println("TorchNet loading in " + this)
    val ref = TorchNet.loadPytorchModel(modelHolder.torchBytes)

    if (weights == null) {
      val w = PytorchModel.getWeightNative(ref).clone()
      weights = Tensor(w, Array(w.length))
    } else {
      PytorchModel.updateWeightNative(ref, weights.storage().array())
    }

    if (gradients == null) {
      gradients = Tensor()
    }
    ref
  }

  override def parameters(): (Array[Tensor[Float]], Array[Tensor[Float]]) = {
    nativeRef
    (Array(weights), Array(gradients))
  }

  override def updateOutput(input: Tensor[Float]): Tensor[Float] = {
    if (this.isTraining()) {
      PytorchModel.updateWeightNative(this.nativeRef, weights.storage().array())
    }

    require(input.isContiguous())
    val data = input.storage().array()
    val size = input.size()
    val offset = input.storageOffset() - 1
    val result = PytorchModel.modelForwardNative(nativeRef, this.isTraining(), data, offset, size)
    val resultTensor = Tensor(result.getData, result.getShape)
    output.set(resultTensor)
  }

  override def updateGradInput(input: Tensor[Float], gradOutput: Tensor[Float]): Tensor[Float] = {
    val data = gradOutput.storage().array()
    val size = gradOutput.size()
    val offset = gradOutput.storageOffset() - 1
    val result = PytorchModel.modelBackwardNative(nativeRef, data, offset, size)
    val resultTensor = Tensor(result.getData, result.getShape)

    gradients.resizeAs(weights)
    val g = PytorchModel.getGradientNative(this.nativeRef)
    System.arraycopy(g, 0, gradients.storage().array(), 0, g.length)
    gradInput.set(resultTensor)
  }

  // TODO: use release if possible. now for larger model it's causing early release
  override def finalize(): Unit = {
    super.finalize()
    PytorchModel.releaseModelNative(nativeRef)
  }
}

object TorchNet {

  PytorchModel.isLoaded
  loadPytorchNatives() // load once per JVM

  private val modelBytesRegistry = new RegistryMap[Array[Byte]]()

  @transient
  private lazy val inDriver = NetUtils.isDriver


  class TorchModelHolder(@transient var torchBytes: Array[Byte], private var id: String)
    extends SerializationHolder {

    override def writeInternal(out: CommonOutputStream): Unit = {
      val (graphDef, _) = modelBytesRegistry.getOrCreate(id) {
        torchBytes
      }
      val len = graphDef.length
      out.writeString(id)
      if (inDriver) {
        out.writeInt(len)
        timing(s"writing ${len / 1024 / 1024}Mb torch model to stream") {
          out.write(graphDef)
        }
      } else {
        out.writeInt(0)
      }
    }

    override def readInternal(in: CommonInputStream): Unit = {
      id = in.readString()
      val (graph, _) = modelBytesRegistry.getOrCreate(id) {
        val len = in.readInt()
        assert(len >= 0, "GraphDef length should be an non-negative integer")
        val graphDef = new Array[Byte](len)
        timing("reading graph def from stream") {
          var numOfBytes = 0
          while (numOfBytes < len) {
            val read = in.read(graphDef, numOfBytes, len - numOfBytes)
            numOfBytes += read
          }
        }
        graphDef
      }

      torchBytes = graph
      id = id
    }

  }

  /**
   * Create a TorchNet from a saved TorchScript Model
   * @param modelPath Path to the TorchScript Model.
   * @return
   */
  def apply(modelPath: String): TorchNet = {
    // TODO: add support for HDFS path
    val modelbytes = Files.readAllBytes(Paths.get(modelPath))
    new TorchNet(new TorchModelHolder(modelbytes, modelPath))
  }

  // extract libs from zoo jar file
  private def loadPytorchNatives(): Unit = {
    loadNativelib("pytorch/libpytorch-engine.so")
  }

  private def loadNativelib(path: String): Unit = {
    val inputStream = TorchNet.getClass.getResourceAsStream(s"/${path}")
    val file = File.createTempFile("PytorchLoader", "tmp")
    val src = Channels.newChannel(inputStream)
    val dest = new FileOutputStream(file).getChannel
    dest.transferFrom(src, 0, Long.MaxValue)
    dest.close()
    src.close()
    val filePath = file.getAbsolutePath
    try {
      System.load(filePath)
    } finally {
      file.delete()
    }
  }


  private[net] def loadPytorchModel(bytes: Array[Byte]): Long = {
    var nativeRef = -1L
    try {
      val tmpFile = File.createTempFile("TorchNet", "_pt")
      Files.write(Paths.get(tmpFile.toURI), bytes)
      nativeRef = PytorchModel.loadModelNative(tmpFile.getAbsolutePath)
      FileUtils.deleteQuietly(tmpFile)
    }
    catch {
      case io: IOException =>
        System.out.println("error during loading Torch model")
        throw io
    }
    nativeRef
  }

}

