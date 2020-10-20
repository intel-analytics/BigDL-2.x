package com.intel.analytics.zoo.tfpark

import java.io.{IOException, ObjectInputStream, ObjectOutputStream}

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.nn.Container
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn.mkldnn.{MklDnnLayer, TensorMMap}
import com.intel.analytics.bigdl.nn.tf.Const
import com.intel.analytics.bigdl.tensor.{QuantizedTensor, QuantizedType, Storage, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.{NumericWildcard, TensorNumeric}
import com.intel.analytics.bigdl.utils.Engine

import com.intel.analytics.bigdl.utils.intermediate.IRGraph
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.EngineRef
import com.intel.analytics.zoo.tfpark.Util._
import org.apache.commons.lang3.SerializationUtils
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

class TFModelBroadcastV2[T: ClassTag]()
                                   (implicit ev: TensorNumeric[T]) extends ModelBroadcast[T] {
//  private type NativeType = (String, (Array[TensorMMap], Array[TensorMMap]))
  private var broadcastModel: Broadcast[ModelInfo[T]] = _
  private var broadcastConsts: Broadcast[Map[String, Tensor[_]]] = _
  private var broadcastParameters: Broadcast[Array[Tensor[T]]] = _
  private var broadcastExtraParameters: Broadcast[Array[Tensor[T]]] = _
//  private var broadcastParametersNative: Broadcast[Array[NativeType]] = _
  private var nodeNumber: Int = _
  private var coreNumber: Int = _

  private def setNodeAndCore(): Unit = {
    nodeNumber = EngineRef.getNodeNumber()
    coreNumber = EngineRef.getCoreNumber()
  }

  /**
    * broadcast the model
    * first get and clear Const values from the model
    * then get and clear the weight and bias parameters from the model
    * finally broadcast Const values, the parameters and model(without parameters) separately
    *
    * @param sc    SparkContext
    * @param model model to broadcast
    * @return this
    */
  override def broadcast(sc: SparkContext, model: Module[T]): this.type = {
    CachedModels.deleteAll(uuid) // delete the models on driver


    // broadcast Consts
//    if (model.isInstanceOf[Container[_, _, T]]) {
//      val moduleConsts = getAndClearConsts(model.asInstanceOf[Container[_, _, T]])
//      // TODO: broadcast Const, model structure and weight in the same broadcast.
//      broadcastConsts = sc.broadcast(moduleConsts)
//    }
    // broadcast weight and model
    val weightsBias = getAndClearWeightBias(model.parameters())
    val extraParams = getAndClearExtraParameters(model.getExtraParameter())
    broadcastModel = sc.broadcast(ModelInfo[T](uuid, model))
    broadcastParameters = sc.broadcast(weightsBias)
    broadcastExtraParameters = sc.broadcast(extraParams)
    var i = 0
    while (i < model.parameters()._1.length){
      println("when broadcast")
      println(s"weights ${i} size: ${model.parameters()._1(i).size().mkString(",")}")
      i += 1
    }
    //      broadcastParameters = sc.broadcast(weightsBias)

    // For quantized model if we don't clone weightsBias, the original model will be released also
    // when we delete all models used in `ModelBroadcast`.
    putWeightBias(weightsBias, model)
    initGradWeightBias(weightsBias, model)
    putExtraParams(extraParams, model)

    setNodeAndCore()
    this
  }

  /**
    * get the broadcast model
    * put the weight and bias back to the model
    *
    * @param initGradient If create a tensor for gradient when fetch the model. Please note that
    *                     the gradient is not needed in model inference
    * @return model
    */
  override def value(initGradient: Boolean = false, shareWeight: Boolean = true): Module[T] = {
    EngineRef.setCoreNumber(coreNumber)
//    Engine.setNodeAndCore(nodeNumber, coreNumber)
    CachedModels.deleteAll(this.uuid)

    val localModel = broadcastModel.value.model.cloneModule()
    val uuid = broadcastModel.value.uuid
    CachedModels.add(uuid, localModel)

    val parameters = if (shareWeight) {
      broadcastParameters.value
    } else {
      SerializationUtils.clone(broadcastParameters.value)
    }

    // share weight
    putWeightBias(parameters, localModel)
//    // share Consts
//    if (localModel.isInstanceOf[Container[_, _, T]] && broadcastConsts.value.nonEmpty) {
//      putConsts(localModel.asInstanceOf[Container[_, _, T]], broadcastConsts.value)
//    }
    // init gradient
    if (initGradient) {
      initGradWeightBias(broadcastParameters.value, localModel)
    }

    putExtraParams(broadcastExtraParameters.value, localModel)


    // share Consts
//    if (localModel.isInstanceOf[Container[_, _, T]] && broadcastConsts.value.nonEmpty) {
//      putConsts(localModel.asInstanceOf[Container[_, _, T]], broadcastConsts.value)
//    }

    localModel
  }
}




