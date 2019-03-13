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

import java.io.{IOException, ObjectOutputStream}

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.mkl._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn._
import com.intel.analytics.bigdl.optim.Regularizer
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
  var wRegularizer: Regularizer[Float] = null,
  var bRegularizer: Regularizer[Float] = null,
  val initWeight: Tensor[Float] = null,
  val initBias: Tensor[Float] = null,
  val initGradWeight: Tensor[Float] = null,
  val initGradBias: Tensor[Float] = null,
  val withBias: Boolean = true,
  val format: DataFormat = DataFormat.NCHW
) extends MklDnnLayer with Initializable with Serializable with MklInt8Convertible {
  private val weightShape = if (nGroup == 1) {
    Array(nOutputPlane, nInputPlane, kernelH, kernelW)
  } else {
    Array (nGroup, nOutputPlane / nGroup, nInputPlane / nGroup, kernelH, kernelW)
  }

  // !!!important!!! this is for weight and input conversion.
  // The weight in forward and updateGradInput is different.
  // The input in updateOutput and accGradParameters is different too.
  // It's `lazy` so the reordermanager need not serialized.
  @transient private lazy val reorderManager = new ReorderManager

  private[mkldnn] val weight = new TensorMMap(weightShape)
  private[mkldnn] val bias = new TensorMMap(Array(nOutputPlane))
  private[mkldnn] val gradWeight = new TensorMMap(weightShape)
  private[mkldnn] val gradBias = new TensorMMap(Array(nOutputPlane))

  // The weight maybe have different format between updateOutput and updateGradInput
  private var weightForBackward: DnnTensor[Float] = _
  private var weightForBackwardMemoryData: MemoryData = _

  // The input maybe have different format between updateOutput and accGradParameters
  private var inputForAcc: DnnTensor[Float] = _
  private var inputForAccMemoryData: MemoryData = _

  @transient private var forwardPrimDesc: Long = 0L

  @transient private var updateOutputMemoryPrimitives: Array[Long] = _
  @transient private var updateOutputTensors: Array[Tensor[Float]] = _
  @transient private var updateGradInputMemoryPrimitives: Array[Long] = _
  @transient private var updateGradInputTensors: Array[Tensor[Float]] = _
  @transient private var updateGradWMemoryPrimitives: Array[Long] = _
  @transient private var updateGradWTensors: Array[Tensor[Float]] = _
  @transient private var paddingTL: Array[Int] = _
  @transient private var paddingBR: Array[Int] = _

  private var _relu = false
  private var _sum = false
  private var _batchNorm = false
  private var _dim = 1
  private var _sumInput = false

  def relu: Boolean = _relu
  def setReLU(value: Boolean = true): this.type = {
    _relu = value
    this
  }

  def batchNorm: Boolean = _batchNorm
  def setBatchNorm(value: Boolean = true): this.type = {
    _batchNorm = value
    this
  }

  def sum: Boolean = _sum
  def setSum(value: Boolean = true): this.type = {
    _sum = value
    this
  }

  var sumOp: MklDnnLayer = null
  def setSumOp(conv: Module[Float], number: Int = 1): this.type = {
    sumOp = conv.asInstanceOf[MklDnnLayer]
    _dim = number
    _sum = true
    this
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
      weightInitMethod.init(weight.dense, if (nGroup == 1) {
        VariableFormat.OUT_IN_KW_KH
      } else {
        VariableFormat.GP_OUT_IN_KW_KH
      })
    } else {
      weight.dense.copy(initWeight)
    }

    if (initBias == null) {
      biasInitMethod.init(bias.dense, VariableFormat.ONE_D)
    } else {
      bias.dense.copy(initBias)
    }
  }

  override private[mkldnn] def initFwdPrimitives(inputs: Array[MemoryData], phase: Phase) = {
    reorderManager.setRuntime(runtime)

    if (_sum && inputs.length > 1) {
      _sumInput = true
      require(inputs.length == 2,
        s"inputs length should be 2 when having sum operation, but get ${inputs.length}")
    }
    val inputMemoryData = inputs(_dim - 1)
    val inputHeight = inputMemoryData.shape(2) // TODO only supports 4-D and nchw
    val inputWidth = inputMemoryData.shape(3)

    val sizes = if (padW == -1 && padH == -1) {
        Utils.getSAMEOutSizeAndPadding(inputHeight, inputWidth, strideH, strideW, kernelH, kernelW)
      } else {
        Utils.getOutSizeAndPadding(inputHeight, inputWidth, strideH, strideW, kernelH, kernelW,
          padH, padW, ceilMode = false)
      }

    val padTop = sizes(0)
    val padBottom = sizes(1)
    val padLeft = sizes(2)
    val padRight = sizes(3)
    val outputHeight = sizes(4)
    val outputWidth = sizes(5)
    paddingTL = Array(padTop, padLeft)
    paddingBR = Array(padBottom, padRight)

    val inputShape = inputMemoryData.shape
    val outputShape = Array(inputMemoryData.shape(0), nOutputPlane, outputHeight, outputWidth)

    val src = NativeData(inputShape, Memory.Format.any)
    val wei = NativeData(weightShape, Memory.Format.any)
    val bis = NativeData(Array(nOutputPlane), Memory.Format.x)
    val dst = NativeData(outputShape, Memory.Format.any)

    val desc = MklDnn.ConvForwardDescInit(
      PropKind.ForwardTraining, AlgKind.ConvolutionDirect,
      src.getMemoryDescription(),
      wei.getMemoryDescription(),
      bis.getMemoryDescription(),
      dst.getMemoryDescription(),
      Array(strideW, strideH), paddingTL, paddingBR,
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

    // The weight on heap should be oihw or goihw format and should reorder it first when using.
    val defaultWeightLayout = if (nGroup == 1) {
      Memory.Format.oihw
    } else {
      Memory.Format.goihw
    }

    weight.setMemoryData(HeapData(weight.dense.size(), defaultWeightLayout),
      realWei, runtime)
    bias.setMemoryData(HeapData(bias.dense.size(), Memory.Format.x),
      bis, runtime)

    weight.sync()
    bias.sync()

    val srcs = Array(realSrc.getPrimitive(runtime), realWei.getPrimitive(runtime),
      bis.getPrimitive(runtime))
    val indexes = Array.fill(srcs.length)(0)
    val dsts = Array(realDst.getPrimitive(runtime))

    val primitive = MklDnn.PrimitiveCreate2(forwardPrimDesc, srcs, indexes, srcs.length,
      dsts, dsts.length)

    updateOutputMemoryPrimitives = srcs ++ dsts
    updateOutputPrimitives = Array(primitive)
    output = initTensor(realDst)

    _inputFormats = if (_sumInput) Array(realSrc, realSrc) else Array(realSrc)
    _outputFormats = Array(realDst)
    (_inputFormats, _outputFormats)
  }

  override def updateOutput(input: Activity): Activity = {
    val inputTensor = if (input.isTensor) {
      input.toTensor[Float]
    } else {
      output = input.toTable.get[Tensor[Float]](3 - _dim).get
      input.toTable.get[Tensor[Float]](_dim).get
    }
    if (updateOutputTensors == null) {
      val buffer = new ArrayBuffer[Tensor[Float]]()
      buffer.append(inputTensor.asInstanceOf[Tensor[Float]])
      buffer.append(weight.native)
      buffer.append(bias.native)
      if (sum && input.isTensor) {
        output = sumOp.output
      }
      buffer.append(output.asInstanceOf[Tensor[Float]])
      updateOutputTensors = buffer.toArray
    }

    updateWithNewTensor(updateOutputTensors, 0, inputTensor)

    if (isTraining()) {
      weight.sync()
      bias.sync()
    }

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
    val bis = NativeData(Array(nOutputPlane), Memory.Format.x)
    val dst = NativeData(outputShape, Memory.Format.any)

    val desc = MklDnn.ConvBackwardDataDescInit(
      AlgKind.ConvolutionDirect,
      src.getMemoryDescription(),
      wei.getMemoryDescription(), // TODO check correctness of strides and padding
      dst.getMemoryDescription(), Array(strideW, strideH), paddingTL, paddingBR,
      MklDnn.PaddingKind.mkldnnPaddingZero)
    val backwardPrimDesc = MklDnn.PrimitiveDescCreate(desc, runtime.engine, forwardPrimDesc)

    val List(realDiffSrc, realWei, realDiffDst) =
      List(Query.DiffSrcPd, Query.WeightsPd, Query.DiffDstPd).map {x =>
        MemoryData.operationWant(backwardPrimDesc, x)
      }

    weightForBackwardMemoryData = realWei

    reorderManager.register(weight.heapData, realWei)

    // computing gradient input doesn't need the input
    val srcs = Array(realDiffDst.getPrimitive(runtime), realWei.getPrimitive(runtime))
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
    // if needed, reorder manager will reorder the wegiht to mkldnn wants
    weightForBackward = reorderManager.infer(Array(weight.heapData),
      Array(weightForBackwardMemoryData), weight.dense).asInstanceOf[DnnTensor[Float]]

    if (updateGradInputTensors == null) {
      val buffer = new ArrayBuffer[Tensor[Float]]()
      buffer.append(gradOutput.asInstanceOf[Tensor[Float]])
      buffer.append(weightForBackward)
      buffer.append(gradInput.asInstanceOf[Tensor[Float]])
      updateGradInputTensors = buffer.toArray
    }

    updateWithNewTensor(updateGradInputTensors, 0, gradOutput)
    updateWithNewTensor(updateGradInputTensors, 1, weightForBackward)

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
    val bis = NativeData(Array(nOutputPlane), Memory.Format.x)

    val desc = MklDnn.ConvBackwardWeightsDescInit(
      AlgKind.ConvolutionDirect,
      src.getMemoryDescription(),
      wei.getMemoryDescription(),
      bis.getMemoryDescription(),
      grad(0).getMemoryDescription(), Array(strideW, strideH), paddingTL, paddingBR,
      MklDnn.PaddingKind.mkldnnPaddingZero)
    val gradWeightPrimDesc = MklDnn.PrimitiveDescCreate(desc, runtime.engine, forwardPrimDesc)

    // TODO here seems some errors ?????? check the realSrc format.
    val List(realSrc, realWei, realDiffDst) =
      List(Query.SrcPd, Query.DiffWeightsPd, Query.DiffDstPd).map { x =>
        MemoryData.operationWant(gradWeightPrimDesc, x)
      }

    // gradient weight should be the same format with weight
    val defaultWeightLayout = if (nGroup == 1) {
      Memory.Format.oihw
    } else {
      Memory.Format.goihw
    }

    gradWeight.setMemoryData(realWei,
      HeapData(gradWeight.dense.size(), defaultWeightLayout), runtime)
    gradBias.setMemoryData(bis,
      HeapData(gradBias.dense.size(), Memory.Format.x), runtime)

    // save the real input format accGradParameters wants, and register the reorder operation
    inputForAccMemoryData = realSrc
    reorderManager.register(inputFormats()(0), realSrc)

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
    // if needed, reorder manager will reorder input to mkldnn wants
    val inputTensor = if (input.isTensor) {
      input.toTensor[Float]
    } else {
      input.toTable.get[Tensor[Float]](_dim).get
    }
    inputForAcc = reorderManager.infer(Array(inputFormats()(0)),
      Array(inputForAccMemoryData), inputTensor).asInstanceOf[DnnTensor[Float]]

    if (updateGradWTensors == null) {
      val buffer = new ArrayBuffer[Tensor[Float]]()
      buffer.append(inputForAcc.asInstanceOf[Tensor[Float]])
      buffer.append(gradOutput.asInstanceOf[Tensor[Float]])
      buffer.append(gradWeight.native)
      buffer.append(gradBias.native)
      updateGradWTensors = buffer.toArray
    }

    updateWithNewTensor(updateGradWTensors, 0, inputForAcc)
    updateWithNewTensor(updateGradWTensors, 1, gradOutput)

    MklDnnOps.streamSubmit(runtime.stream, 1, accGradientPrimitives,
      accGradientPrimitives.length, updateGradWMemoryPrimitives, updateGradWTensors)

    gradWeight.sync()
    gradBias.sync()

    if (null != wRegularizer) {
      wRegularizer.accRegularization(weight.dense, gradWeight.dense, scaleW)
    }
    if (withBias && null != bRegularizer) {
      bRegularizer.accRegularization(bias.dense, gradBias.dense, scaleB)
    }
  }

  override def parameters(): (Array[Tensor[Float]], Array[Tensor[Float]]) = {
    if (withBias) {
      (Array(weight.dense, bias.dense), Array(gradWeight.dense, gradBias.dense))
    } else {
      (Array(weight.dense), Array(gradWeight.dense))
    }

  }

  // we need not implement it, because the grad parameters will clean by mkldnn
  override def zeroGradParameters(): Unit = {
  }

  override def release(): Unit = {
    super.release()
    List(weight, bias, gradWeight, gradBias).foreach(_.release())
    if (weightForBackward != null) { weightForBackward.release() }
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
    wRegularizer: Regularizer[Float] = null,
    bRegularizer: Regularizer[Float] = null,
    initWeight: Tensor[Float] = null,
    initBias: Tensor[Float] = null,
    initGradWeight: Tensor[Float] = null,
    initGradBias: Tensor[Float] = null,
    withBias: Boolean = true,
    format: DataFormat = DataFormat.NCHW): SpatialConvolution = {
    new SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW,
      dH, padW, padH, nGroup, propagateBack, wRegularizer, bRegularizer,
      initWeight, initBias, initGradWeight, initGradBias, withBias, format)
  }
}
