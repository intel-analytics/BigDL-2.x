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

import java.io.{File, IOException}
import java.nio.file.{Files, Paths}

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractCriterion, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}
import com.intel.analytics.zoo.pipeline.api.net.TorchNet.TorchModelHolder
import org.apache.commons.io.FileUtils

import scala.reflect.ClassTag

class TorchCriterion private(private val lossHolder: TorchModelHolder)
    extends AbstractCriterion[Activity, Activity, Float]() {

  implicit val ev = TensorNumeric.NumericFloat
  implicit val tag: ClassTag[Float] = ClassTag.Float

  /**
   * sequential id in cpp: std::vector<std::shared_ptr<torch::jit::script::Module>> handles;
   * mark the model as transient and reload TorchNet from byteArray on executors
   */
  @transient
  lazy val nativeRef: Long = {
    println("TorchCriterion loading in " + this)
    val ref = TorchCriterion.loadPytorchCriterion(lossHolder.torchBytes)
    ref
  }

  override def updateOutput(input: Activity, target: Activity): Float = {
    val inputTabel = if (input.isTensor) T(input.toTensor) else input.toTable
    val targetTable = if (target.isTensor) T(target.toTensor) else target.toTable

    val (sto1, off1, shape1) = TorchCriterion.extract(inputTabel)
    val (sto2, off2, shape2) = TorchCriterion.extract(targetTable)

    val result = PytorchModelWrapper.lossForwardNative(nativeRef, sto1, off1,
      shape1, sto2, off2, shape2)
    Tensor(result.getData, result.getShape).mean()
  }

  override def updateGradInput(input: Activity, target: Activity): Activity = {

    val result = PytorchModelWrapper.lossBackwardNative(nativeRef)
    if (result.length == 1) {
      val resultTensor = Tensor(result(0).getData, result(0).getShape)
      if (gradInput == null) {
        gradInput = Tensor()
      }
      gradInput.toTensor.set(resultTensor)
    } else {
      if (gradInput == null) {
        gradInput = T()
      }
      gradInput.toTable.clear()
      result.foreach { t =>
        gradInput.toTable.insert(Tensor(t.getData, t.getShape))
      }
    }
    gradInput
  }

  override def finalize(): Unit = {
    super.finalize()
    PytorchModel.releaseLossNative(nativeRef)
  }

}

object TorchCriterion {

  /**
   * Create a TorchCriterion from a saved TorchScript Model
   * @param lossPath Path to the TorchScript Model.
   * @return
   */
  def apply(lossPath: String): TorchCriterion = {
    // TODO: add support for HDFS path
    val modelbytes = Files.readAllBytes(Paths.get(lossPath))
    new TorchCriterion(new TorchModelHolder(modelbytes, lossPath))
  }

  private[net] def loadPytorchCriterion(bytes: Array[Byte]): Long = {
    var nativeRef = -1L
    try {
      val tmpFile = File.createTempFile("TorchNet", "_pt")
      Files.write(Paths.get(tmpFile.toURI), bytes)
      nativeRef = PytorchModel.loadLossNative(tmpFile.getAbsolutePath)
      FileUtils.deleteQuietly(tmpFile)
    } catch {
      case io: IOException =>
        System.out.println("error during loading Torch model")
        throw io
    }
    nativeRef
  }


  private[net] def extract(t: Table): (Array[Array[Float]], Array[Int], Array[Array[Int]]) = {
    val tensors = t.toSeq[Tensor[Float]]

    val tuples = tensors.map { t =>
      require(t.isContiguous())
      (t.storage(), t.storageOffset() - 1, t.size())
    }
    val storages = tuples.map(_._1.array()).toArray
    val offsets = tuples.map(_._2).toArray
    val shapes = tuples.map(_._3).toArray
    (storages, offsets, shapes)
  }

}
