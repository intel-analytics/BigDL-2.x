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
import java.util.zip.ZipInputStream

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractCriterion, AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.pipeline.api.Predictable
import com.intel.analytics.zoo.pipeline.api.net.TorchNet.TorchModelHolder
import org.apache.commons.io.FileUtils

import scala.reflect.ClassTag

/**
 * [[TorchNet]] wraps a TorchScript model as a single layer.
 */
class TorchNet private(private val modelHolder: TorchModelHolder,
                       // TODO: separate it to TorchCriterion?
                       private val lossHolder: TorchModelHolder)
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
    val ref = if (lossHolder != null)
      TorchNet.loadPytorchModel(modelHolder.torchBytes, lossHolder.torchBytes)
    else
      TorchNet.loadPytorchModel(modelHolder.torchBytes, modelHolder.torchBytes)

    if (weights == null) {
      val w = PytorchModel.getWeightNative(ref).clone()
      weights = Tensor(w, Array(w.length))
      gradients = Tensor(Array.tabulate(w.length)(i => 0f), Array(w.length))
    } else{
      PytorchModel.updateWeightNative(ref, weights.storage().array())
    }
    ref
  }

  ModelBroadcast


  override def parameters(): (Array[Tensor[Float]], Array[Tensor[Float]]) = {
    nativeRef
    println("-------- weights: " + weights.storage().array().take(5).mkString(", ")
      + "-------- gradients: " + gradients.storage().array().take(5).mkString(", ")
      + "-------- TORCH_NET: " + this
      + "-------- TORCH_MODEL: " + this.nativeRef)
    (Array(weights), Array(gradients))
  }

  override def updateOutput(input: Tensor[Float]): Tensor[Float] = {
    if(this.isTraining()) {
      PytorchModel.updateWeightNative(this.nativeRef, weights.storage().array())
    }

    require(input.isContiguous())
    val data = input.storage().array()
    val size = input.size()
    val offset = input.storageOffset() - 1
    val result = PytorchModel.forwardNative(nativeRef, data, offset, size)
    val resultTensor = Tensor(result.getData, result.getShape)
    output.set(resultTensor)
  }

  override def updateGradInput(input: Tensor[Float], gradOutput: Tensor[Float]): Tensor[Float] = {
    val data = gradOutput.storage().array()
    val size = gradOutput.size()
    val offset = gradOutput.storageOffset() - 1
    val result = PytorchModel.backwardNative(nativeRef, data, offset, size)
    val resultTensor = Tensor(result.getData, result.getShape)

    val g = PytorchModel.getGradientNative(this.nativeRef).clone()
    System.arraycopy(g, 0, gradients.storage().array(), 0, g.length)

    gradInput.set(resultTensor)
  }

  override def accGradParameters(input: Tensor[Float], gradOutput: Tensor[Float]): Unit =  {
    super.accGradParameters(input, gradOutput)
  }

  override def release(): Unit = {
    super.release()
    if (nativeRef != null) {
      PytorchModel.releaseNative(nativeRef)
    }
  }
}

object TorchNet {

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
   * @param path Path to the TorchScript Model.
   * @return
   */
  def apply(path: String): TorchNet = {
    // TODO: add support for HDFS path
    val modelbytes = Files.readAllBytes(Paths.get(path))
    new TorchNet(new TorchModelHolder(modelbytes, path), null)
  }

  def apply(modelPath: String, lossPath: String): TorchNet = {
    // TODO: add support for HDFS path
    val modelbytes = Files.readAllBytes(Paths.get(modelPath))
    val lossbytes = Files.readAllBytes(Paths.get(lossPath))
    new TorchNet(new TorchModelHolder(modelbytes, modelPath),
      new TorchModelHolder(lossbytes, lossPath))
  }

  // extract libs from zoo jar file
  private def loadPytorchNatives(): Unit = {
    val tmpDir = com.google.common.io.Files.createTempDir()
    val libStream = TorchNet.getClass.getResourceAsStream(s"/pytorch/lib/libtorch-shared-with-deps-latest.zip")
    unzip(libStream, tmpDir)

    try {
      System.load(tmpDir + "/libtorch/lib/libtorch.so")
      loadNative("pytorch/libpytorch-engine.so")
    }
    finally {
      FileUtils.deleteDirectory(tmpDir)
    }
  }

  private def loadNative(path: String): Unit = {
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

  private def unzip(inputStream: InputStream, outputPath: File) {
    try {
      val buffer = new Array[Byte](2048)
      val istream = new ZipInputStream(inputStream)
      var entry = istream.getNextEntry
      while (entry != null) {
        val entryDestination = new File(outputPath, entry.getName)
        if (entry.isDirectory) entryDestination.mkdirs
        else {
          entryDestination.getParentFile.mkdirs
          val fos = new FileOutputStream(entryDestination)
          val bos = new BufferedOutputStream(fos, buffer.length)
          var len = istream.read(buffer)
          while (len > 0) {
            bos.write(buffer, 0, len)
            len = istream.read(buffer)
          }
          bos.close()
        }
        entry = istream.getNextEntry
      }
    } catch {
      case io: IOException =>
        System.out.println("error during loading loading pytorch libs")
        throw io
    }
  }

  private[net] def loadPytorchModel(bytes: Array[Byte], lossBytes: Array[Byte]): Long = {
    var nativeRef = -1L
    try {
      val tmpFile = File.createTempFile("TorchNet", "_pt")
      Files.write(Paths.get(tmpFile.toURI), bytes)
      val lossFile = File.createTempFile("TorchNet", "_lpt")
      Files.write(Paths.get(lossFile.toURI), lossBytes)
      nativeRef = PytorchModel.loadNative(tmpFile.getAbsolutePath, lossFile.getAbsolutePath)
      FileUtils.deleteQuietly(tmpFile)
      FileUtils.deleteQuietly(lossFile)
    }
    catch {
      case io: IOException => {
        System.out.println("error during loading Torch model")
        throw io
      }
    }
    nativeRef
  }

}

class TorchIdentityCriterion extends AbstractCriterion[Activity, Activity, Float]() {

  override def updateOutput(input: Activity, target: Activity): Float = {
    input.toTensor[Float].mean()
  }
  override def updateGradInput(input: Activity, target: Activity): Activity = {
    gradInput = target
    gradInput
  }
}

