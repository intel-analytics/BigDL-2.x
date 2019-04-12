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


package com.intel.analytics.zoo.pipeline.api.autograd

import com.intel.analytics.bigdl.nn.abstractnn._
import com.intel.analytics.bigdl.nn.keras.{KerasLayer, KerasLayerSerializable}
import com.intel.analytics.bigdl.nn.tf.InternalWithoutInput
import com.intel.analytics.bigdl.nn.{InitializationMethod, RandomUniform, VariableFormat}
import com.intel.analytics.bigdl.serialization.Bigdl.{AttrValue, BigDLModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.serializer.converters.DataConverter
import com.intel.analytics.bigdl.utils.serializer.converters.DataConverter.ArrayConverter
import com.intel.analytics.bigdl.utils.serializer._
import com.intel.analytics.bigdl.utils.{Node, Shape}
import com.intel.analytics.zoo.models.seq2seq.RNNEncoder._
import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.api.keras.layers.Recurrent
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.KerasUtils

import scala.reflect.ClassTag
import scala.reflect.runtime._

private[zoo] class KerasParameter[T: ClassTag] private[zoo](val inputShape: Shape,
    val initMethod: InitializationMethod = RandomUniform(-0.05, 0.05),
    val initWeight: Tensor[T] = null,
    val trainable: Boolean = true)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Activity, Activity, T](inputShape)
    with InternalWithoutInput with Net {

  def setWeight(tensor: Tensor[T]): Unit = {
    this.labor.asInstanceOf[InternalParameter[T]].setWeight(tensor)
  }

  def getWeight(): Tensor[T] = {
    this.labor.asInstanceOf[InternalParameter[T]].weight
  }

  override def computeOutputShape(inputShape: Shape): Shape = inputShape

  override def doBuild(inputShape: Shape): AbstractModule[Activity, Activity, T] =
    new InternalParameter[T](
      shape = inputShape,
      initMethod = this.initMethod,
      initWeight = this.initWeight,
      trainable = this.trainable).asInstanceOf[AbstractModule[Activity, Activity, T]]

  override def skipDuplicateCheck(): Boolean = true

//  override def clearState() : this.type = {
//    output = Tensor[T]()
//    gradInput = Tensor[T]()
//    this
//  }
}

//object KerasParameter extends KerasLayerSerializable {
//
//    override def doSerializeModule[T: ClassTag](context: SerializeContext[T],
//                                                parameterBuilder : BigDLModule.Builder)
//                                               (implicit ev: TensorNumeric[T]) : Unit = {
////      super.doSerializeModule(context, parameterBuilder)
//      val parameter = context.moduleData.module.asInstanceOf[KerasParameter[T]]
//      val labor = context.moduleData.module.
//        asInstanceOf[KerasLayer[Activity, Activity, T]].labor
//
//      val shapeBuilder = AttrValue.newBuilder
//      DataConverter.setAttributeValue(context, shapeBuilder,
//              parameter.inputShape, universe.typeOf[Shape])
//      parameterBuilder.putAttr("shape", shapeBuilder.build)
//
//      val initMBuilder = AttrValue.newBuilder
//      DataConverter.setAttributeValue(context, initMBuilder,
//        parameter.initMethod, universe.typeOf[InitializationMethod])
//      parameterBuilder.putAttr("initM", initMBuilder.build)
//
//      val trainableBuilder = AttrValue.newBuilder
//      DataConverter.setAttributeValue(context, trainableBuilder,
//        parameter.trainable, universe.typeOf[Boolean])
//      parameterBuilder.putAttr("trainable", trainableBuilder.build)
//
//      val initWBuilder = AttrValue.newBuilder
//      DataConverter.setAttributeValue(context, initWBuilder,
//        parameter.initWeight, ModuleSerializer.tensorType)
//      parameterBuilder.putAttr("initW", initWBuilder.build)
//
//      val weightBuilder = AttrValue.newBuilder
//      DataConverter.setAttributeValue(context, weightBuilder,
//              parameter.getWeight(), ModuleSerializer.tensorType)
//      parameterBuilder.putAttr("weight", weightBuilder.build)
//
//          appendKerasLabel(context, parameterBuilder)
//    }
//
//    override def doLoadModule[T: ClassTag](context : DeserializeContext)
//                                          (implicit ev: TensorNumeric[T]) : AbstractModule[Activity, Activity, T] = {
//
//      val attrMap = context.bigdlModule.getAttrMap
//
//      val shapeAttr = attrMap.get("shape")
//      val shape = DataConverter.getAttributeValue(context, shapeAttr).asInstanceOf[Shape]
//
//      val initMAttr = attrMap.get("initM")
//      val initM = DataConverter.getAttributeValue(context, initMAttr).
//        asInstanceOf[InitializationMethod]
//
//      val initWAttr = attrMap.get("initW")
//      val initW = DataConverter.getAttributeValue(context, initWAttr).
//        asInstanceOf[Tensor[T]]
//
//      val trainableAttr = attrMap.get("trainable")
//      val trainable = DataConverter.getAttributeValue(context, trainableAttr).
//        asInstanceOf[Boolean]
//
//      val weightAttr = attrMap.get("weight")
//      val weight = DataConverter.getAttributeValue(context, weightAttr).
//        asInstanceOf[Tensor[T]]
//
//          val parameter = new KerasParameter[T](shape, initM, initW, trainable)
//
//      parameter.setWeight(weight)
//
//      parameter.asInstanceOf[AbstractModule[Activity, Activity, T]]
//    }
//}

/**
 * Parameters is trainable Variable and it can be treated as a constant value
 * if trainable is set to be False.
 * @param inputShape Shape of this Parameter
 * @param initMethod A method used to initialize the Parameter.
 *                   The default value is RandomUniform(-0.05, 0.05)
 * @param initWeight The init value for the Parameter
 * @param trainable It's true by default, meaning the value would be updated by gradient.
 */
class Parameter[T: ClassTag] private[zoo](val inputShape: Shape,
    val initMethod: InitializationMethod = RandomUniform(-0.05, 0.05),
    val initWeight: Tensor[T] = null,
    val trainable: Boolean = true,
    name: String = null)(
    implicit ev: TensorNumeric[T])
  extends Variable[T](null, name) {

  // build and init the KerasParameter
  val kerasParameter = new KerasParameter[T](inputShape, initMethod, initWeight, trainable)
  this.node = new Node(kerasParameter)
  this.node.element.asInstanceOf[KerasParameter[T]].build(inputShape)

  def setWeight(tensor: Tensor[T]): Unit = {
    this.node.element.asInstanceOf[KerasParameter[T]].setWeight(tensor)
  }

  def getWeight(): Tensor[T] = {
    this.node.element.asInstanceOf[KerasParameter[T]].getWeight()
  }
}

object Parameter {

  /**
   * Create a trainable Variable
   */
  def apply[T: ClassTag](
      inputShape: Shape,
      initMethod: InitializationMethod = RandomUniform(-0.05, 0.05),
      initWeight: Tensor[T] = null,
      trainable: Boolean = true,
      name: String = null)(implicit ev: TensorNumeric[T]): Parameter[T] = {
    new Parameter[T](inputShape, initMethod = initMethod, initWeight = initWeight,
      trainable = trainable, name = name)
  }

//  override def doSerializeModule[T: ClassTag](context: SerializeContext[T],
//                                              parameterBuilder : BigDLModule.Builder)
//                                             (implicit ev: TensorNumeric[T]) : Unit = {
//
//    val parameter = context.moduleData.module.asInstanceOf[Parameter[T]]
//
//    val shapeBuilder = AttrValue.newBuilder
//    DataConverter.setAttributeValue(context, shapeBuilder,
//            parameter.inputShape, universe.typeOf[Shape])
//    parameterBuilder.putAttr("shape", shapeBuilder.build)
//
//    val initMBuilder = AttrValue.newBuilder
//    DataConverter.setAttributeValue(context, initMBuilder,
//      parameter.initMethod, universe.typeOf[InitializationMethod])
//    parameterBuilder.putAttr("initM", initMBuilder.build)
//
//    val trainableBuilder = AttrValue.newBuilder
//    DataConverter.setAttributeValue(context, trainableBuilder,
//      parameter.trainable, universe.typeOf[Boolean])
//    parameterBuilder.putAttr("trainable", trainableBuilder.build)
//
//    val initWBuilder = AttrValue.newBuilder
//    DataConverter.setAttributeValue(context, initWBuilder,
//      parameter.initWeight, universe.typeOf[Tensor[_]])
//    parameterBuilder.putAttr("initW", initWBuilder.build)
//
//    val weightBuilder = AttrValue.newBuilder
//    DataConverter.setAttributeValue(context, weightBuilder,
//            parameter.kerasParameter.getWeight(), universe.typeOf[Tensor[_]])
//    parameterBuilder.putAttr("weight", weightBuilder.build)
//
//        appendKerasLabel(context, parameterBuilder)
//  }
//
//  override def doLoadModule[T: ClassTag](context : DeserializeContext)
//                                        (implicit ev: TensorNumeric[T]) : AbstractModule[Activity, Activity, T] = {
//
//    val attrMap = context.bigdlModule.getAttrMap
//
//    val shapeAttr = attrMap.get("shape")
//    val shape = DataConverter.getAttributeValue(context, shapeAttr).asInstanceOf[Shape]
//
//    val initMAttr = attrMap.get("initM")
//    val initM = DataConverter.getAttributeValue(context, initMAttr).
//      asInstanceOf[InitializationMethod]
//
//    val initWAttr = attrMap.get("initW")
//    val initW = DataConverter.getAttributeValue(context, initWAttr).
//      asInstanceOf[Tensor[T]]
//
//    val trainableAttr = attrMap.get("trainable")
//    val trainable = DataConverter.getAttributeValue(context, trainableAttr).
//      asInstanceOf[Boolean]
//
//    val weightAttr = attrMap.get("weight")
//    val weight = DataConverter.getAttributeValue(context, weightAttr).
//      asInstanceOf[Tensor[T]]
//
//        val parameter = Parameter[T](shape, initM, initW, trainable)
//
//    parameter.setWeight(weight)
//
//    parameter.asInstanceOf[AbstractModule[Activity, Activity, T]]
//  }
}

private[zoo] class InternalParameter[T: ClassTag](
    val shape: Shape,
    val initMethod: InitializationMethod = RandomUniform(-0.05, 0.05),
    val initWeight: Tensor[T] = null,
    val trainable: Boolean = true)(implicit ev: TensorNumeric[T])
  extends TensorModule[T] with Initializable {

  var weight: Tensor[T] =
    if (initWeight != null) initWeight else Tensor[T](shape.toSingle().toArray)

  val gradWeight: Tensor[T] = Tensor[T]()

  setInitMethod(weightInitMethod = initMethod)

  def setWeight(tensor: Tensor[T]): Unit = {
    this.weight = tensor
  }

  override def reset(): Unit = {
    if (initWeight == null) {
      weightInitMethod.init(weight, VariableFormat.OUT_IN)
    }
    zeroGradParameters()
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output = weight.clone()
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput = gradOutput.clone()
    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    gradWeight.resizeAs(gradOutput)
    if (trainable) {
      gradWeight.copy(gradOutput)
    } else {
      gradWeight.fill(ev.zero)
    }
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (Array(this.weight), Array(this.gradWeight))
  }

//  override def clearState() : this.type = {
//    output = Tensor[T]()
//    gradInput = Tensor[T]()
//    this
//  }

  override def equals(other: Any): Boolean = {
    if (!other.isInstanceOf[InternalParameter[_]]) return false
    this.eq(other.asInstanceOf[InternalParameter[_]])
  }

  override def hashCode(): Int = System.identityHashCode(this)
}

//object InternalParameter extends ModuleSerializable {
////  ModuleSerializer.registerModule(
////    "com.intel.analytics.zoo.pipeline.api.autograd.InternalParameter",
////    InternalParameter)
//
//  override def doSerializeModule[T: ClassTag](context: SerializeContext[T],
//    parameterBuilder : BigDLModule.Builder)
//   (implicit ev: TensorNumeric[T]) : Unit = {
//
//    val parameter = context.moduleData.module.asInstanceOf[InternalParameter[T]]
//
//    val shapeBuilder = AttrValue.newBuilder
//    DataConverter.setAttributeValue(context, shapeBuilder,
////      parameter.inputShape, universe.typeOf[Shape])
//      parameter.shape, universe.typeOf[Shape])
//    parameterBuilder.putAttr("shape", shapeBuilder.build)
//
//    val initMBuilder = AttrValue.newBuilder
//    DataConverter.setAttributeValue(context, initMBuilder,
//      parameter.initMethod, universe.typeOf[InitializationMethod])
//    parameterBuilder.putAttr("initM", initMBuilder.build)
//
//    val trainableBuilder = AttrValue.newBuilder
//    DataConverter.setAttributeValue(context, trainableBuilder,
//      parameter.trainable, universe.typeOf[Boolean])
//    parameterBuilder.putAttr("trainable", trainableBuilder.build)
//
//    val initWBuilder = AttrValue.newBuilder
//    DataConverter.setAttributeValue(context, initWBuilder,
//      parameter.initWeight, ModuleSerializer.tensorType)
//    parameterBuilder.putAttr("initW", initWBuilder.build)
//
//    val weightBuilder = AttrValue.newBuilder
//    DataConverter.setAttributeValue(context, weightBuilder,
////      parameter.kerasParameter.getWeight(), universe.typeOf[Tensor[_]])
//      parameter.weight, ModuleSerializer.tensorType)
//    parameterBuilder.putAttr("weight", weightBuilder.build)
//
////    appendKerasLabel(context, parameterBuilder)
//  }
//
//  override def doLoadModule[T: ClassTag](context : DeserializeContext)
//    (implicit ev: TensorNumeric[T]) : AbstractModule[Activity, Activity, T] = {
//
//    val attrMap = context.bigdlModule.getAttrMap
//
//    val shapeAttr = attrMap.get("shape")
//    val shape = DataConverter.getAttributeValue(context, shapeAttr).asInstanceOf[Shape]
//
//    val initMAttr = attrMap.get("initM")
//    val initM = DataConverter.getAttributeValue(context, initMAttr).
//      asInstanceOf[InitializationMethod]
//
//    val initWAttr = attrMap.get("initW")
//    val initW = DataConverter.getAttributeValue(context, initWAttr).
//      asInstanceOf[Tensor[T]]
//
//    val trainableAttr = attrMap.get("trainable")
//    val trainable = DataConverter.getAttributeValue(context, trainableAttr).
//      asInstanceOf[Boolean]
//
//    val weightAttr = attrMap.get("weight")
//    val weight = DataConverter.getAttributeValue(context, weightAttr).
//      asInstanceOf[Tensor[T]]
//
////    val parameter = Parameter[T](shape, initM, initW, trainable)
//val parameter = new InternalParameter[T](shape, initM, initW, trainable)
//    parameter.setWeight(weight)
//
//    parameter.asInstanceOf[AbstractModule[Activity, Activity, T]]
//  }
//}

private[zoo] class InternalConstant[T: ClassTag](val data: Tensor[T])
  (implicit ev: TensorNumeric[T])
  extends TensorModule[T] {

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output = data
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    null
  }

  override def equals(other: Any): Boolean = {
    if (!other.isInstanceOf[InternalConstant[_]]) return false
    this.eq(other.asInstanceOf[InternalConstant[_]])
  }

  override def hashCode(): Int = System.identityHashCode(this)
}

private[zoo] class KerasConstant[T: ClassTag] private[zoo](
  val data: Tensor[T]
  )(implicit ev: TensorNumeric[T])
  extends KerasLayer[Activity, Activity, T]()
    with InternalWithoutInput with Net {

  override def doBuild(inputShape: Shape): AbstractModule[Activity, Activity, T] =
    new InternalConstant[T](data).asInstanceOf[AbstractModule[Activity, Activity, T]]

  override def skipDuplicateCheck(): Boolean = true

  override def computeOutputShape(inputShape: Shape): Shape = Shape(data.size())
}

class Constant[T: ClassTag] private[zoo](val data: Tensor[T],
  name: String = null)(implicit ev: TensorNumeric[T])
  extends Variable[T](null, name) {

  this.node = new Node(new KerasConstant[T](data))
  this.node.element.asInstanceOf[KerasLayer[Tensor[T], Tensor[T], T]].build(Shape(data.size()))
}
