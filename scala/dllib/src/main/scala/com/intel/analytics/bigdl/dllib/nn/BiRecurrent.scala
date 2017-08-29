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

package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, TensorModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.serializer.{ContainerSerializable, DataConverter, ModuleData, ModuleSerializer}
import com.intel.analytics.bigdl.utils.{T, Table}
import serialization.Bigdl.{AttrValue, BigDLModule}

import scala.reflect.ClassTag
import scala.reflect.runtime._

/**
 * This layer implement a bidirectional recurrent neural network
 * @param merge concat or add the output tensor of the two RNNs. Default is add
 * @param ev numeric operator
 * @tparam T numeric type
 */
class BiRecurrent[T : ClassTag] (
  private val merge: AbstractModule[Table, Tensor[T], T] = null,
  val batchNormParams: BatchNormParams[T] = null,
  val isSplitInput: Boolean = false)
  (implicit ev: TensorNumeric[T]) extends Container[Tensor[T], Tensor[T], T] {

  val timeDim = 2
  val featDim = 3
  val layer: Recurrent[T] = Recurrent[T](batchNormParams)
  val revLayer: Recurrent[T] = Recurrent[T](batchNormParams)

  private var birnn = Sequential[T]()

  if (isSplitInput) {
    birnn.add(BifurcateSplitTable[T](featDim))
  } else {
    birnn.add(ConcatTable()
      .add(Identity[T]())
      .add(Identity[T]()))
  }

  birnn
    .add(ParallelTable[T]()
      .add(layer)
      .add(Sequential[T]()
        .add(Reverse[T](timeDim))
        .add(revLayer)
        .add(Reverse[T](timeDim))))
  if (merge == null) birnn.add(CAddTable[T](true))
  else birnn.add(merge)

  override def add(module: AbstractModule[_ <: Activity, _ <: Activity, T]):
    BiRecurrent.this.type = {
    layer.add(module)
    revLayer.add(module.cloneModule())
    this
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output = birnn.updateOutput(input).toTensor[T]
    output
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    birnn.accGradParameters(input, gradOutput)
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput = birnn.updateGradInput(input, gradOutput).toTensor[T]
    gradInput
  }

  /**
   * This function returns two arrays. One for the weights and the other the gradients
   * Custom modules should override this function if they have parameters
   *
   * @return (Array of weights, Array of grad)
   */
  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = birnn.parameters()

  override def updateParameters(learningRate: T): Unit = birnn.updateParameters(learningRate)

  /**
   * If the module has parameters, this will zero the accumulation of the gradients with respect
   * to these parameters. Otherwise, it does nothing.
   */
  override def zeroGradParameters(): Unit = birnn.zeroGradParameters()

  override def training(): BiRecurrent.this.type = {
    super.training()
    birnn.training()
    this
  }

  override def evaluate(): BiRecurrent.this.type = {
    super.evaluate()
    birnn.evaluate()
    this
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[BiRecurrent[T]]


  /**
   * Clear cached activities to save storage space or network bandwidth. Note that we use
   * Tensor.set to keep some information like tensor share
   *
   * The subclass should override this method if it allocate some extra resource, and call the
   * super.clearState in the override method
   *
   * @return
   */
  override def clearState(): BiRecurrent.this.type = {
    birnn.clearState()
    this
  }

  override def toString(): String = s"${getPrintName}($timeDim, $birnn)"

  override def equals(other: Any): Boolean = other match {
    case that: BiRecurrent[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        timeDim == that.timeDim &&
        layer == that.layer &&
        revLayer == that.revLayer &&
        birnn == that.birnn
    case _ => false
  }

  override def hashCode(): Int = {
    val state = Seq(super.hashCode(), timeDim, layer, revLayer, birnn)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }
}

object BiRecurrent extends ContainerSerializable {
  def apply[@specialized(Float, Double) T: ClassTag](
    merge: AbstractModule[Table, Tensor[T], T] = null,
    batchNormParams: BatchNormParams[T] = null,
    isSplitInput: Boolean = false)
    (implicit ev: TensorNumeric[T]) : BiRecurrent[T] = {
    new BiRecurrent[T](merge, batchNormParams, isSplitInput)
  }

  override def doLoadModule[T: ClassTag](model : BigDLModule)
    (implicit ev: TensorNumeric[T]) : AbstractModule[Activity, Activity, T] = {

    val attrMap = model.getAttrMap

    val merge = DataConverter.
      getAttributeValue(attrMap.get("merge")).
      asInstanceOf[AbstractModule[Table, Tensor[T], T]]

    val isSplitInput = DataConverter
      .getAttributeValue(attrMap.get("isSplitInput"))
      .asInstanceOf[Boolean]

    val flag = DataConverter
      .getAttributeValue(attrMap.get("bnorm"))
      .asInstanceOf[Boolean]

    val biRecurrent = if (flag) {
      BiRecurrent(merge, batchNormParams = BatchNormParams(), isSplitInput = isSplitInput)
    } else {
      BiRecurrent(merge, isSplitInput = isSplitInput)
    }

    biRecurrent.birnn = DataConverter.
      getAttributeValue(attrMap.get("birnn")).
      asInstanceOf[Sequential[T]]

    if (flag) {
      val bnormEpsAttr = attrMap.get("bnormEps")
      biRecurrent.batchNormParams.eps =
        DataConverter.getAttributeValue(bnormEpsAttr)
          .asInstanceOf[Double]

      val bnormMomentumAttr = attrMap.get("bnormMomentum")
      biRecurrent.batchNormParams.momentum =
        DataConverter.getAttributeValue(bnormMomentumAttr)
          .asInstanceOf[Double]

      val bnormInitWeightAttr = attrMap.get("bnormInitWeight")
      biRecurrent.batchNormParams.initWeight =
        DataConverter.getAttributeValue(bnormInitWeightAttr)
          .asInstanceOf[Tensor[T]]

      val bnormInitBiasAttr = attrMap.get("bnormInitBias")
      biRecurrent.batchNormParams.initBias =
        DataConverter.getAttributeValue(bnormInitBiasAttr)
          .asInstanceOf[Tensor[T]]

      val bnormInitGradWeightAttr = attrMap.get("bnormInitGradWeight")
      biRecurrent.batchNormParams.initGradWeight =
        DataConverter.getAttributeValue(bnormInitGradWeightAttr)
          .asInstanceOf[Tensor[T]]

      val bnormInitGradBiasAttr = attrMap.get("bnormInitGradBias")
      biRecurrent.batchNormParams.initGradBias =
        DataConverter.getAttributeValue(bnormInitGradBiasAttr)
          .asInstanceOf[Tensor[T]]
    }

    createBigDLModule(model, biRecurrent)

    biRecurrent

  }

  override def doSerializeModule[T: ClassTag](module : ModuleData[T],
                                            birecurrentBuilder : BigDLModule.Builder)
                                           (implicit ev: TensorNumeric[T]) : Unit = {

    val birecurrentModule = module.module.
      asInstanceOf[BiRecurrent[T]]

    val mergeBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(mergeBuilder,
      birecurrentModule.merge,
      ModuleSerializer.tensorModuleType)
    birecurrentBuilder.putAttr("merge", mergeBuilder.build)

    val isSplitInputBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(isSplitInputBuilder,
      birecurrentModule.isSplitInput,
      universe.typeOf[Boolean])
    birecurrentBuilder.putAttr("isSplitInput", isSplitInputBuilder.build)

    val birnnBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(birnnBuilder,
      birecurrentModule.birnn,
      ModuleSerializer.tensorModuleType)
    birecurrentBuilder.putAttr("birnn", birnnBuilder.build)

    val flag = if (birecurrentModule.batchNormParams != null) {

      println("save batchNormParams")
      val bnormEpsBuilder = AttrValue.newBuilder
      DataConverter.setAttributeValue(bnormEpsBuilder,
        birecurrentModule.batchNormParams.eps, universe.typeOf[Double])
      birecurrentBuilder.putAttr("bnormEps", bnormEpsBuilder.build)

      val bnormMomentumBuilder = AttrValue.newBuilder
      DataConverter.setAttributeValue(bnormMomentumBuilder,
        birecurrentModule.batchNormParams.momentum, universe.typeOf[Double])
      birecurrentBuilder.putAttr("bnormMomentum", bnormMomentumBuilder.build)

      val bnormInitWeightBuilder = AttrValue.newBuilder
      DataConverter.setAttributeValue(bnormInitWeightBuilder,
        birecurrentModule.batchNormParams.initWeight, ModuleSerializer.tensorType)
      birecurrentBuilder.putAttr("bnormInitWeight", bnormInitWeightBuilder.build)

      val bnormInitBiasBuilder = AttrValue.newBuilder
      DataConverter.setAttributeValue(bnormInitBiasBuilder,
        birecurrentModule.batchNormParams.initBias, ModuleSerializer.tensorType)
      birecurrentBuilder.putAttr("bnormInitBias", bnormInitBiasBuilder.build)

      val bnormInitGradWeightBuilder = AttrValue.newBuilder
      DataConverter.setAttributeValue(bnormInitGradWeightBuilder,
        birecurrentModule.batchNormParams.initGradWeight, ModuleSerializer.tensorType)
      birecurrentBuilder.putAttr("bnormInitGradWeight", bnormInitGradWeightBuilder.build)

      val bnormInitGradBiasBuilder = AttrValue.newBuilder
      DataConverter.setAttributeValue(bnormInitGradBiasBuilder,
        birecurrentModule.batchNormParams.initGradBias, ModuleSerializer.tensorType)
      birecurrentBuilder.putAttr("bnormInitGradBias", bnormInitGradBiasBuilder.build)

      true
    } else {
      false
    }

    val bNormBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(bNormBuilder,
      flag, universe.typeOf[Boolean])
    birecurrentBuilder.putAttr("bnorm", bNormBuilder.build)

    createSerializeBigDLModule(birecurrentBuilder, module)
  }
}
