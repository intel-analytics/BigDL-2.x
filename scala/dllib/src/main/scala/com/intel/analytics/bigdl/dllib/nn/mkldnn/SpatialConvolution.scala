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

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.mkl._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn._
import com.intel.analytics.bigdl.tensor.{DnnTensor, Tensor}

import scala.collection.mutable.ArrayBuffer

class SpatialConvolution(
  val nInputPlane: Int,
  val nOutputPlane: Int,
  val kernelW: Int,
  val kernelH: Int,
  val strideW: Int = 1,
  val strideH: Int = 1,
  val padW: Int = 0,
  val padH: Int = 0,
  val nGroup: Int = 1,
  val propagateBack: Boolean = true,
  val initWeight: Tensor[Float] = null,
  val initBias: Tensor[Float] = null,
  val initGradWeight: Tensor[Float] = null,
  val initGradBias: Tensor[Float] = null,
  val withBias: Boolean = true,
  val format: DataFormat = DataFormat.NCHW
) extends MklDnnLayer with Initializable {
  private val weightShape = if (nGroup == 1) {
    Array(nOutputPlane, nInputPlane, kernelH, kernelW)
  } else {
    Array (nGroup, nOutputPlane / nGroup, nInputPlane / nGroup, kernelH, kernelW)
  }

  // !!!important!!! this is for weight conversion. The weights in forward and backward is
  // different.
  val reorderManager = new ReorderManager

  val weight: DnnTensor[Float] = DnnTensor[Float](weightShape)
  var weightForBackward: DnnTensor[Float] = _
  val bias: DnnTensor[Float] = DnnTensor[Float](Array(nOutputPlane))
  val gradWeight: DnnTensor[Float] = DnnTensor[Float](weightShape)
  val gradBias: DnnTensor[Float] = DnnTensor[Float](Array(nOutputPlane))

  var forwardPrimDesc: Long = 0L

  var updateOutputMemoryPrimitives: Array[Long] = _
  var updateOutputTensors: Array[Tensor[Float]] = _
  var updateGradInputMemoryPrimitives: Array[Long] = _
  var updateGradInputTensors: Array[Tensor[Float]] = _
  var updateGradWMemoryPrimitives: Array[Long] = _
  var updateGradWTensors: Array[Tensor[Float]] = _

  var _relu = false
  var _sum = false

  def relu: Boolean = _relu
  def setReLU(value: Boolean = true): this.type = {
    _relu = value
    this
  }

  def sum: Boolean = _sum
  def setSum(value: Boolean = true): this.type = {
    _sum = value
    this
  }

  var sumOp: MklDnnLayer = null
  def setSumOp(conv: Module[Float]): this.type = {
    sumOp = conv.asInstanceOf[MklDnnLayer]
    this
  }

  object ParamsShape {
    var weight: MemoryData = _
    var weightForBackward: MemoryData = _
    var bias: MemoryData = _
    var gradWeight: MemoryData = _
    var gradBias: MemoryData = _
  }

  private def getOutputShape(oh: Int, ow: Int, batchSize: Int = -1): Array[Int] = {
    format match {
      case DataFormat.NCHW =>
        if (batchSize == -1) {
          Array(nOutputPlane, oh, ow)
        } else {
          Array(batchSize, nOutputPlane, oh, ow)
        }
      case DataFormat.NHWC =>
        if (batchSize == -1) {
          Array(oh, ow, nOutputPlane)
        } else {
          Array(batchSize, oh, ow, nOutputPlane)
        }

    }
  }

  {
    val stdv = 1.0 / math.sqrt(kernelW * kernelH * nInputPlane)
    val wInit: InitializationMethod = RandomUniform(-stdv, stdv)
    val bInit: InitializationMethod = if (withBias) RandomUniform(-stdv, stdv)
    else null
    setInitMethod(wInit, bInit)
  }

  override def reset(): Unit = {
    if (initWeight == null) { // TODO only support oihw format weights
      val t = Tensor[Float](weightShape)
      weightInitMethod.init(t, VariableFormat.OUT_IN)
      weight.copy(t)
    } else {
      weight.copy(initWeight)
    }

    if (initBias == null) {
      val t = Tensor[Float](Array(nOutputPlane))
      biasInitMethod.init(t, VariableFormat.ONE_D)
      bias.copy(t)
    } else {
      bias.copy(initBias)
    }

    zeroGradParameters()
  }

  override private[mkldnn] def initFwdPrimitives(inputs: Array[MemoryData], phase: Phase) = {
    reorderManager.setRuntime(runtime)

    val inputHeight = inputs(0).shape(2) // TODO only supports 4-D and nchw
    val inputWidth = inputs(0).shape(3)

    val sizes = if (padW == -1 && padH == -1) {
        Utils.getSAMEOutSizeAndPadding(inputHeight, inputWidth, strideH, strideW, kernelH, kernelW)
      } else {
        Utils.getOutSizeAndPadding(inputHeight, inputWidth, strideH, strideW, kernelH, kernelW,
          padH, padW, ceilMode = false)
      }

    val outputHeight = sizes(4)
    val outputWidth = sizes(5)

    val inputShape = inputs(0).shape
    val outputShape = Array(inputs(0).shape(0), nOutputPlane, outputHeight, outputWidth)

    val src = NativeData(inputShape, Memory.Format.any)
    val wei = NativeData(weightShape, Memory.Format.any)
    val bis = NativeData(bias.size(), Memory.Format.x)
    val dst = NativeData(outputShape, Memory.Format.any)

    val desc = MklDnn.ConvForwardDescInit(
      PropKind.ForwardTraining, AlgKind.ConvolutionDirect,
      src.getMemoryDescription(),
      wei.getMemoryDescription(),
      bis.getMemoryDescription(),
      dst.getMemoryDescription(),
      Array(strideW, strideH), Array(padH, padW), Array(padH, padW), // TODO check the meaning
      MklDnn.PaddingKind.mkldnnPaddingZero)

    forwardPrimDesc = if (relu || sum) {
      val postOps = MklDnn.CreatePostOps()
      if (sum) {
        MklDnn.PostOpsAppendSum(postOps, 1.0f)
      }
      if (relu) {
        MklDnn.PostOpsAppendEltwise(postOps, 1.0f, AlgKind.EltwiseRelu, 0.0f, 0.0f)
      }
      val attr = MklDnn.CreateAttr()
      MklDnn.AttrSetPostOps(attr, postOps)

      MklDnn.PrimitiveDescCreateV2(desc, attr, runtime.engine, 0)
      // TODO we should destroy these ops
    } else {
      MklDnn.PrimitiveDescCreate(desc, runtime.engine, 0)
    }

    val List(realSrc, realWei, realDst) = List(Query.SrcPd, Query.WeightsPd, Query.DstPd).map {x =>
      MemoryData.operationWant(forwardPrimDesc, x)
    }

    ParamsShape.weight = realWei
    ParamsShape.bias = bis

    val srcs = Array(realSrc.getPrimitive(runtime), realWei.getPrimitive(runtime),
      bis.getPrimitive(runtime))
    val indexes = Array.fill(srcs.length)(0)
    val dsts = Array(realDst.getPrimitive(runtime))

    val primitive = MklDnn.PrimitiveCreate2(forwardPrimDesc, srcs, indexes, srcs.length,
      dsts, dsts.length)

    updateOutputMemoryPrimitives = srcs ++ dsts
    updateOutputPrimitives = Array(primitive)
    output = initTensor(dst)

    _inputFormats = Array(realSrc)
    _outputFormats = Array(realDst)
    (_inputFormats, _outputFormats)
  }

  override def updateOutput(input: Activity): Activity = {
    if (updateOutputTensors == null) {
      val buffer = new ArrayBuffer[Tensor[Float]]()
      buffer.append(input.asInstanceOf[Tensor[Float]])
      buffer.append(weight)
      buffer.append(bias)
      if (sum) {
        output = sumOp.output
      }
      buffer.append(output.asInstanceOf[Tensor[Float]])
      updateOutputTensors = buffer.toArray
    }

    updateWithNewTensor(updateOutputTensors, 0, input)

    MklDnnOps.streamSubmit(runtime.stream, 1, updateOutputPrimitives, updateOutputPrimitives.length,
      updateOutputMemoryPrimitives, updateOutputTensors)

    output
  }

  override private[mkldnn] def initBwdPrimitives(grad: Array[MemoryData], phase: Phase) = {
    val inputShape = inputFormats()(0).shape.length match {
      case 1 => inputFormats()(0).shape ++ Array(1) // TODO Test
      case _ => inputFormats()(0).shape
    }

    val outputShape = outputFormats()(0).shape

    val src = NativeData(inputShape, Memory.Format.any)
    val wei = NativeData(weightShape, Memory.Format.any)
    val bis = NativeData(bias.size(), Memory.Format.x)
    val dst = NativeData(outputShape, Memory.Format.any)

    val desc = MklDnn.ConvBackwardDataDescInit(
      AlgKind.ConvolutionDirect,
      src.getMemoryDescription(),
      wei.getMemoryDescription(), // TODO check correctness of strides and padding
      dst.getMemoryDescription(), Array(strideW, strideH), Array(padH, padW), Array(padH, padW),
      MklDnn.PaddingKind.mkldnnPaddingZero)
    val backwardPrimDesc = MklDnn.PrimitiveDescCreate(desc, runtime.engine, forwardPrimDesc)

    val List(realDiffSrc, realWei, realDiffDst) =
      List(Query.DiffSrcPd, Query.WeightsPd, Query.DiffDstPd).map {x =>
        MemoryData.operationWant(backwardPrimDesc, x)
      }

    ParamsShape.weightForBackward = realWei

    reorderManager.register(ParamsShape.weight, realWei)

    val srcs = Array(realDiffDst.getPrimitive(runtime), realWei.getPrimitive(runtime),
      inputFormats()(0).getPrimitive(runtime))
    val indexes = Array.fill(srcs.length)(0)
    val dsts = Array(realDiffSrc.getPrimitive(runtime))

    val primitive = MklDnn.PrimitiveCreate2(backwardPrimDesc, srcs, indexes, srcs.length,
      dsts, dsts.length)

    updateGradInputMemoryPrimitives = srcs ++ dsts
    updateGradInputPrimitives = Array(primitive)
    gradInput = initTensor(realDiffSrc)

    _gradInputFormats = Array(realDiffSrc)
    _gradOutputFormats = Array(realDiffDst)
    (_gradOutputFormats, _gradInputFormats)
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    weightForBackward = reorderManager.infer(Array(ParamsShape.weight),
      Array(ParamsShape.weightForBackward), weight).asInstanceOf[DnnTensor[Float]]

    if (updateGradInputTensors == null) {
      val buffer = new ArrayBuffer[Tensor[Float]]()
      buffer.append(gradOutput.asInstanceOf[Tensor[Float]])
      buffer.append(weightForBackward)
      buffer.append(input.asInstanceOf[Tensor[Float]])
      buffer.append(gradInput.asInstanceOf[Tensor[Float]])
      updateGradInputTensors = buffer.toArray
    }

    updateWithNewTensor(updateGradInputTensors, 2, input)
    updateWithNewTensor(updateGradInputTensors, 0, gradOutput)

    MklDnnOps.streamSubmit(runtime.stream, 1, updateGradInputPrimitives,
      updateGradInputPrimitives.length, updateGradInputMemoryPrimitives, updateGradInputTensors)

    gradInput
  }
  override private[mkldnn] def initGradWPrimitives(grad: Array[MemoryData],
    phase: Phase): Array[MemoryData] = {
    val inputShape = inputFormats()(0).shape
    val outputShape = inputFormats()(0).shape

    val src = NativeData(inputShape, Memory.Format.any)
    val wei = NativeData(weightShape, Memory.Format.any)
    val bis = NativeData(bias.size(), Memory.Format.x)

    val desc = MklDnn.ConvBackwardWeightsDescInit(
      AlgKind.ConvolutionDirect,
      src.getMemoryDescription(),
      wei.getMemoryDescription(),
      bis.getMemoryDescription(),
      grad(0).getMemoryDescription(), Array(strideW, strideH), Array(padH, padW), Array(padH, padW),
      MklDnn.PaddingKind.mkldnnPaddingZero)
    val gradWeightPrimDesc = MklDnn.PrimitiveDescCreate(desc, runtime.engine, forwardPrimDesc)

    // TODO here seems some errors ?????? check the realSrc format.
    val List(realSrc, realWei, realDiffDst) =
      List(Query.SrcPd, Query.DiffWeightsPd, Query.DiffDstPd).map { x =>
        MemoryData.operationWant(gradWeightPrimDesc, x)
      }

    ParamsShape.gradWeight = realWei
    ParamsShape.gradBias = bis

    val srcs = Array(realSrc.getPrimitive(runtime), realDiffDst.getPrimitive(runtime))
    val indexes = Array.fill(srcs.length)(0)
    val dsts = Array(realWei.getPrimitive(runtime), bis.getPrimitive(runtime))

    val primitive = MklDnn.PrimitiveCreate2(gradWeightPrimDesc, srcs, indexes, srcs.length,
      dsts, dsts.length)

    updateGradWMemoryPrimitives = srcs ++ dsts
    accGradientPrimitives = Array(primitive)

    _gradOutputFormatsForWeight = Array(realDiffDst)
    (_gradOutputFormatsForWeight)
  }

  override def accGradParameters(input: Activity, gradOutput: Activity): Unit = {
    if (updateGradWTensors == null) {
      val buffer = new ArrayBuffer[Tensor[Float]]()
      buffer.append(input.asInstanceOf[Tensor[Float]])
      buffer.append(gradOutput.asInstanceOf[Tensor[Float]])
      buffer.append(gradWeight)
      buffer.append(gradBias)
      updateGradWTensors = buffer.toArray
    }

    updateWithNewTensor(updateGradWTensors, 0, input)
    updateWithNewTensor(updateGradWTensors, 1, gradOutput)

    MklDnnOps.streamSubmit(runtime.stream, 1, accGradientPrimitives,
      accGradientPrimitives.length, updateGradWMemoryPrimitives, updateGradWTensors)
  }

  override def parameters(): (Array[Tensor[Float]], Array[Tensor[Float]]) = {
    (Array(weight, bias), Array(gradWeight, gradBias))
  }

  override def zeroGradParameters(): Unit = {
    gradWeight.zero()
    gradBias.zero()
  }

  override def parametersWithShape(): (Array[MemoryData], Array[MemoryData]) = {
    (Array(ParamsShape.weight, ParamsShape.bias), Array(ParamsShape.gradWeight, ParamsShape.bias))
  }
}

object SpatialConvolution {
  def apply(
    nInputPlane: Int,
    nOutputPlane: Int,
    kW: Int,
    kH: Int,
    dW: Int = 1,
    dH: Int = 1,
    padW: Int = 0,
    padH: Int = 0,
    nGroup: Int = 1,
    propagateBack: Boolean = true,
    initWeight: Tensor[Float] = null,
    initBias: Tensor[Float] = null,
    initGradWeight: Tensor[Float] = null,
    initGradBias: Tensor[Float] = null,
    withBias: Boolean = true,
    format: DataFormat = DataFormat.NCHW): SpatialConvolution = {
    new SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW,
      dH, padW, padH, nGroup, propagateBack,
      initWeight, initBias, initGradWeight, initGradBias, withBias, format)
  }
}
