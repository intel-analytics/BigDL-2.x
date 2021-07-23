/*
 * Copyright 2016 The BigDL Authors.
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

package com.intel.analytics.bigdl.nn.mkldnn

import com.intel.analytics.bigdl.mkl.{DataType, Memory, MklDnn, PropKind, Query, Stream => DnnStream}
import com.intel.analytics.bigdl.nn
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn.mkldnn.Phase.{InferencePhase, TrainingPhase}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{DenseType, Tensor}
import com.intel.analytics.bigdl.utils.Shape

import scala.collection.mutable.ArrayBuffer

class SoftMax(val axis: Int = -1) extends MklDnnLayer {
  @transient private var updateOutputTensors: Array[Tensor[Float]] = _
  @transient private var updateOutputMemoryPrimitives: Array[Long] = _
  @transient private var updateGradInputTensors: Array[Tensor[Float]] = _
  @transient private var updateGradInputMemoryPrimitives: Array[Long] = _
  @transient private var modelPhase: Phase = null

  private var defaultAxis = 0

  private def initPhase(phase: Phase): Unit = {
    if (phase != null) return modelPhase = phase
    isTraining() match {
      case true =>
        modelPhase = TrainingPhase
      case false =>
        modelPhase = InferencePhase
    }
  }

  private def format(shape: Array[Int]): Int = {
    shape.length match {
      case 2 => Memory.Format.nc
      case 4 => Memory.Format.nchw
      case _ => throw new UnsupportedOperationException(s"${getName()} unsupported input shape")
    }
  }

  override private[mkldnn] def initFwdPrimitives(inputs: Array[MemoryData], phase: Phase) = {
    initPhase(phase)
    defaultAxis = inputs(0).shape.length match {
      case 1 => 0
      case 2 => 1
      case 3 => 0
      case 4 => 1
      case _ => throw new UnsupportedOperationException("1D, 2D, 3D or 4D tensor expected")
    }

    _inputFormats = Array(NativeData(inputs(0).shape, inputs(0).layout, DataType.F32))

    val localInputFormat = if (inputs(0).shape.length == 3 &&
      inputs(0).layout == Memory.Format.ntc) {
      // note: here, the format and the true memory layout is not consistent.
      // for ntc input, we should reshape the `shape` and make the format to tnc
      val shape = Array(inputs(0).shape(1), inputs(0).shape(0), inputs(0).shape(2))
      NativeData(shape, Memory.Format.tnc)
    } else {
      _inputFormats(0)
    }

    val desc = MklDnnMemory.SoftMaxForwardDescInit(PropKind.Forward,
      localInputFormat.getMemoryDescription(), if (axis == -1) defaultAxis else axis)
    val forwardPrimDesc = MklDnnMemory.PrimitiveDescCreate(desc, runtime.engine, 0L)

    _outputFormats = if (inputs(0).shape.length ==3 &&
      inputs(0).layout == Memory.Format.ntc) {
      // because set the input format as tnc first, we should set the output to ntc.
      Array(NativeData(inputs(0).shape, Memory.Format.ntc))
    } else {
      Array(MemoryData.primitiveOutput(forwardPrimDesc))
    }

    val srcs = Array(inputs(0).getPrimitive(runtime))
    val indexes = Array(0)
    val dsts = Array(_outputFormats(0).getPrimitive(runtime))

    val primitive = MklDnnMemory.PrimitiveCreate2(forwardPrimDesc, srcs, indexes,
      srcs.length, dsts, dsts.length)

    updateOutputPrimitives = Array(primitive)
    updateOutputMemoryPrimitives = srcs ++ dsts

    output = initTensor(_outputFormats(0))

    (_inputFormats, _outputFormats)
  }

  override private[mkldnn] def initBwdPrimitives(grad: Array[MemoryData], phase: Phase) = {
    val desc = MklDnnMemory.SoftMaxBackwardDescInit(PropKind.Backward,
      grad(0).getMemoryDescription(), outputFormats()(0).getMemoryDescription(),
      if (axis == -1) defaultAxis else axis)
    val primDesc = MklDnnMemory.PrimitiveDescCreate(desc, runtime.engine, 0L)

    _gradOutputFormats = grad.clone()
    _gradInputFormats = Array(MemoryData.operationWant(primDesc, Query.DiffSrcPd))

    val srcs = Array(outputFormats()(0).getPrimitive(runtime), grad(0).getPrimitive(runtime))
    val indexes = Array(0)
    val dsts = Array(_gradInputFormats(0).getPrimitive(runtime))

    val primitive = MklDnnMemory.PrimitiveCreate2(primDesc, srcs, indexes,
      srcs.length, dsts, dsts.length)

    updateGradInputPrimitives = Array(primitive)
    updateGradInputMemoryPrimitives = srcs ++ dsts

    gradInput = initTensor(_gradInputFormats(0))

    (_gradInputFormats, _gradOutputFormats)
  }

  override def updateOutput(input: Activity): Activity = {
    if (updateOutputTensors == null) {
      val buffer = new ArrayBuffer[Tensor[Float]]()
      buffer.append(input.asInstanceOf[Tensor[Float]])
      buffer.append(output.asInstanceOf[Tensor[Float]])
      updateOutputTensors = buffer.toArray
    }

    input.toTensor[Float].getTensorType match {
      case DenseType => updateOutputTensors(0) = input.toTensor
      case _ =>
    }

    MklDnnOps.streamSubmit(runtime.stream, 1,
      updateOutputPrimitives,
      updateOutputPrimitives.length,
      updateOutputMemoryPrimitives, updateOutputTensors)
    output
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    if (updateGradInputTensors == null) {
      val buffer = new ArrayBuffer[Tensor[Float]]()
      buffer.append(output.asInstanceOf[Tensor[Float]])
      buffer.append(gradOutput.asInstanceOf[Tensor[Float]])
      buffer.append(gradInput.asInstanceOf[Tensor[Float]])

      updateGradInputTensors = buffer.toArray
    }

    gradOutput.toTensor[Float].getTensorType match {
      case DenseType => updateGradInputTensors(1) = gradOutput.toTensor
      case _ =>
    }

    MklDnnOps.streamSubmit(runtime.stream, 1,
      updateGradInputPrimitives,
      updateGradInputPrimitives.length,
      updateGradInputMemoryPrimitives, updateGradInputTensors
    )

    gradInput
  }

  override def computeOutputShape(inputShape: Shape): Shape = {
    inputShape
  }
}

object SoftMax {
  def apply(axis: Int = -1)(implicit ev: TensorNumeric[Float]): SoftMax = {
    new SoftMax(axis)
  }
}
