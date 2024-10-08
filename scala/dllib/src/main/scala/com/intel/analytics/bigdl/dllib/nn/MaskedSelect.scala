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
package com.intel.analytics.bigdl.dllib.nn

import com.intel.analytics.bigdl.dllib.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.utils.Table
import com.intel.analytics.bigdl.dllib.utils.serializer._
import com.intel.analytics.bigdl.dllib.utils.serializer.converters.DataConverter
import com.intel.analytics.bigdl.serialization.Bigdl.{AttrValue, BigDLModule}

import scala.reflect.ClassTag


/**
 * Performs a torch.MaskedSelect on a Tensor. The mask is supplied as a tabular argument
 * with the input on the forward and backward passes.
 */

@SerialVersionUID(8596309896021196822L)
class MaskedSelect[T: ClassTag]
(implicit ev: TensorNumeric[T]) extends AbstractModule[Table, Tensor[T], T]{

  private val maskIndices = Tensor[T]()
  private val maskIndexBuffer = Tensor[T]()
  private val gradBuffer = Tensor[T]()
  private val gradMask = Tensor[T]()

  override def updateOutput(input: Table): Tensor[T] = {
    val inputTensor = input[Tensor[T]](1)
    val mask = input[Tensor[T]](2)
    if (ev.toType[Double](mask.sum()) > 0) inputTensor.maskedSelect(mask, output)
    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[T]): Table = {
    val inputTensor = input[Tensor[T]](1)
    val mask = input[Tensor[T]](2)

    // ignore CudaTensor
    maskIndexBuffer.range(1, mask.nElement())
    maskIndexBuffer.resizeAs(mask)

    if (ev.toType[Double](mask.sum()) > 0) maskIndexBuffer.maskedSelect(mask, maskIndices)

    gradBuffer.resize(inputTensor.nElement()).zero()
    gradBuffer.scatter(1, maskIndices, gradOutput)
    gradBuffer.resizeAs(inputTensor)

    gradInput.insert(1, gradBuffer)
    gradInput.insert(2, gradMask.resizeAs(mask).zero())
    gradInput
  }

  override def clearState() : this.type = {
    super.clearState()
    maskIndices.set()
    maskIndexBuffer.set()
    gradBuffer.set()
    gradMask.set()
    this
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[MaskedSelect[T]]

  override def equals(other: Any): Boolean = other match {
    case that: MaskedSelect[T] =>
      super.equals(that) &&
        (that canEqual this)
    case _ => false
  }

  override def hashCode(): Int = {
    def getHashCode(a: Any): Int = if (a == null) 0 else a.hashCode()
    val state = Seq(super.hashCode())
    state.map(getHashCode).foldLeft(0)((a, b) => 31 * a + b)
  }
}

object MaskedSelect extends ModuleSerializable {
  def apply[@specialized(Float, Double) T: ClassTag]()
      (implicit ev: TensorNumeric[T]) : MaskedSelect[T] = {
    new MaskedSelect[T]()
  }

  override def doLoadModule[T: ClassTag](context: DeserializeContext)
    (implicit ev: TensorNumeric[T]) : AbstractModule[Activity, Activity, T] = {
    val maskedSelect = super.doLoadModule(context).asInstanceOf[MaskedSelect[T]]
    val modelAttributes = context.bigdlModule.getAttrMap

    val maskIndices = modelAttributes.get("maskIndices")
    val maskIndicesValue = DataConverter.getAttributeValue(context, maskIndices)
      .asInstanceOf[Tensor[T]]
    maskedSelect.maskIndices.resizeAs(maskIndicesValue).copy(maskIndicesValue)

    val maskIndexBuffer = modelAttributes.get("maskIndexBuffer")
    val maskIndexBufferValue = DataConverter.getAttributeValue(context, maskIndexBuffer)
      .asInstanceOf[Tensor[T]]
    maskedSelect.maskIndexBuffer.resizeAs(maskIndexBufferValue).copy(maskIndexBufferValue)

    val gradBufferBuffer = modelAttributes.get("gradBuffer")
    val gradBufferValue = DataConverter.getAttributeValue(context, gradBufferBuffer)
      .asInstanceOf[Tensor[T]]
    maskedSelect.gradBuffer.resizeAs(gradBufferValue).copy(gradBufferValue)

    val gradMaskBuffer = modelAttributes.get("gradMask")
    val gradMaskValue = DataConverter.getAttributeValue(context, gradMaskBuffer)
      .asInstanceOf[Tensor[T]]
    maskedSelect.gradMask.resizeAs(gradMaskValue).copy(gradMaskValue)

    maskedSelect
  }

  override def doSerializeModule[T: ClassTag](context: SerializeContext[T],
                                              maskedSelectBuilder : BigDLModule.Builder)
                                           (implicit ev: TensorNumeric[T]) : Unit = {
    val masKSelect = context.moduleData.module.asInstanceOf[MaskedSelect[T]]

    val maskIndicesBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, maskIndicesBuilder, masKSelect.maskIndices,
      ModuleSerializer.tensorType)
    maskedSelectBuilder.putAttr("maskIndices", maskIndicesBuilder.build)

    val maskIndexBufferBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, maskIndexBufferBuilder, masKSelect.maskIndexBuffer,
      ModuleSerializer.tensorType)
    maskedSelectBuilder.putAttr("maskIndexBuffer", maskIndexBufferBuilder.build)

    val gradBufferBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, gradBufferBuilder, masKSelect.gradBuffer,
      ModuleSerializer.tensorType)
    maskedSelectBuilder.putAttr("gradBuffer", gradBufferBuilder.build)

    val gradMaskBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, gradMaskBuilder, masKSelect.gradMask,
      ModuleSerializer.tensorType)
    maskedSelectBuilder.putAttr("gradMask", gradMaskBuilder.build)

  }
}
