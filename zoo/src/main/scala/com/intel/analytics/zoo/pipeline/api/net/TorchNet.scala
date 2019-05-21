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

import java.nio.file.{Files, Paths}

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.pipeline.api.Predictable
import org.apache.commons.io.FileUtils

import scala.reflect.ClassTag

/**
 * [[TorchNet]] wraps a TorchScript model as a single layer.
 */
class TorchNet private(private val path: String)
  extends AbstractModule[Tensor[Float], Tensor[Float], Float] with Predictable[Float] {

  /**
   * binary content of model in {path}, used as model serialization during broadcast.
   */
  private val modelbytes = Files.readAllBytes(Paths.get(path))

  /**
   * mark the model as transient and reload TorchNet from byteArray on executors
   */
  @transient
  private lazy val torchModel = {
    TorchNet.load(modelbytes)
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

  loadPytorch() // run once per JVM

  /**
   * Create a TorchNet from a saved TorchScript Model
   * @param path Path to the TorchScript Model.
   * @return
   */
  def apply(path: String): TorchNet = {
    //TODO: add support for HDFS path
    new TorchNet(path)
  }

  // extract libs from zoo jar file
  private def loadPytorch(): Unit = {
    val tmpDir = com.google.common.io.Files.createTempDir()
    val libStream = TorchNet.getClass.getResourceAsStream(s"/pytorch/lib.zip")
    TARCompress.unzip(libStream, tmpDir)

    try {
      System.load(tmpDir + "/libtorch.so")
      System.load(tmpDir + "/libpytorch-engine.so")
    }
    finally {
      FileUtils.deleteDirectory(tmpDir)
    }
  }

  private def load(bytes: Array[Byte]): PytorchModel = {
    new PytorchModel().load(bytes)
  }

}