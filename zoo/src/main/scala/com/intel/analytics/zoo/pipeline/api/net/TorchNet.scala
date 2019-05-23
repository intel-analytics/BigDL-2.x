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
import java.nio.file.{Files, Paths}
import java.util.zip.ZipInputStream

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.pipeline.api.Predictable
import com.intel.analytics.zoo.pipeline.api.net.TorchNet.TorchModelHolder
import org.apache.commons.io.FileUtils

import scala.collection.mutable
import scala.reflect.ClassTag

/**
 * [[TorchNet]] wraps a TorchScript model as a single layer.
 */
class TorchNet private(private val modelHolder: TorchModelHolder)
  extends AbstractModule[Tensor[Float], Tensor[Float], Float] with Predictable[Float] {

  /**
   * mark the model as transient and reload TorchNet from byteArray on executors
   */
  @transient
  private lazy val torchModel = {
    TorchNet.load(modelHolder.torchBytes)
  }

  override def parameters(): (Array[Tensor[Float]], Array[Tensor[Float]]) = {
    (Array.empty, Array.empty)
  }

  protected val module: Module[Float] = this
  implicit val ev = TensorNumeric.NumericFloat
  implicit val tag: ClassTag[Float] = ClassTag.Float

  override def updateOutput(input: Tensor[Float]): Tensor[Float] = {
    require(input.isContiguous())
    val data = input.storage().array()
    val size = input.size()
    val offset = input.storageOffset()
    val result = torchModel.forward(data, offset, size)
    val resultTensor  = Tensor(result.getData, result.getShape)
    output.set(resultTensor)
  }

  override def updateGradInput(input: Tensor[Float], gradOutput: Tensor[Float]): Tensor[Float] = {
    throw new NotImplementedError("backward is not supported for now")
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
    //TODO: add support for HDFS path
    val modelbytes = Files.readAllBytes(Paths.get(path))
    new TorchNet(new TorchModelHolder(modelbytes, path))
  }

  // extract libs from zoo jar file
  private def loadPytorchNatives(): Unit = {
    val tmpDir = com.google.common.io.Files.createTempDir()
    val libStream = TorchNet.getClass.getResourceAsStream(s"/pytorch/lib.zip")
    unzip(libStream, tmpDir)

    try {
      System.load(tmpDir + "/libtorch.so")
      System.load(tmpDir + "/libpytorch-engine.so")
    }
    finally {
      FileUtils.deleteDirectory(tmpDir)
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

  private def load(bytes: Array[Byte]): PytorchModel = {
    new PytorchModel().load(bytes)
  }

}