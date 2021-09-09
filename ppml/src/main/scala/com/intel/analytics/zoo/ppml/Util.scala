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

package com.intel.analytics.zoo.ppml

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.Tensor

import com.intel.analytics.zoo.ppml.generated.FLProto._
import org.apache.log4j.Logger

import scala.collection.JavaConverters._
import scala.reflect.ClassTag
import scala.util.Random


object Util {

  private val logger = Logger.getLogger(getClass)

  def toFloatTensor(data: Array[Float], shape: Array[Int]): FloatTensor = {
    FloatTensor
      .newBuilder()
      .addAllTensor(data.map(float2Float).toIterable.asJava)
      .addAllShape(shape
        .map(int2Integer).toIterable.asJava)
      .build()
  }
  def toFloatTensor(t: Tensor[Float]): FloatTensor = {
    FloatTensor
      .newBuilder()
      .addAllTensor(t.toTensor[Float].contiguous().storage()
        .array().slice(t.storageOffset() - 1, t.storageOffset() - 1 + t.nElement())
        .map(float2Float).toIterable.asJava)
      .addAllShape(t.toTensor[Float].size()
        .map(int2Integer).toIterable.asJava)
      .build()
  }
  def toFloatTensor(data: Array[Float]): FloatTensor = {
    toFloatTensor(data, Array(data.length))
  }
//  def uploadModel(client: ClientGrpc, model: Module[Float], flVersion: Int): Unit = {
//    val parameterTable = model.getParametersTable()
//
//    val metadata = ParameterServerServiceInterface.TableMetaData.newBuilder
//      .setName("test").setVersion(flVersion).build
//
//    val weights = getParametersFromModel(model)._1
//
//    val tensor =
//      ParameterServerServiceInterface.FloatTensor.newBuilder()
//        .addAllTensor(weights.storage.toList.map(v => float2Float(v)))
//        .addAllShape(weights.size.toList.map(v => int2Integer(v)))
//        .build()
//
//
//    val metamodel = ParameterServerServiceInterface.Table.newBuilder
//      .putTable("weights", tensor)
//      .setMetaData(metadata)
//      .build
//    client.uploadData(metamodel)
//  }
//
//  def uploadModelByNode(client: ClientGrpc, model: Module[Float]): Unit = {
//    val parameterTable = model.getParametersTable()
//
//    val metadata = ParameterServerServiceInterface.TableMetaData.newBuilder
//      .setName(model.getName).setVersion(0).build
//
//    val tensorMap = collection.mutable.Map[String, ParameterServerServiceInterface.FloatTensor]()
//    parameterTable.keySet.map { name =>
//      val params = parameterTable[Btable](name)
//      params.keySet.map { key =>
//        if (key == "weightBias" || key == "weight") {
//          val tensor = params[Tensor[Float]](key)
//          val dataArr = tensor.storage.toList.map(v => float2Float(v))
//          val shape = tensor.size.toList.map(v => int2Integer(v))
//          val tensorMessage = ParameterServerServiceInterface
//            .FloatTensor.newBuilder.addAllTensor(dataArr)
//            .addAllShape(shape).build
//          tensorMap += (name + "_" + key -> tensorMessage)
//        }
//      }
//    }
//
//    val metamodel = ParameterServerServiceInterface.Table.newBuilder
//      .putAllTable(mapAsJavaMap(tensorMap))
//      .setMetaData(metadata)
//      .build
//    client.uploadData(metamodel)
//  }
//
//  def updateModel(model: Module[Float],
//                  modelData: ParameterServerServiceInterface.Table): Unit = {
//    val weigthBias = modelData.getTableMap.get("weights")
//    val data = weigthBias.getTensorList.asScala.map(v => Float2float(v)).toArray
//    val shape = weigthBias.getShapeList.asScala.map(v => Integer2int(v)).toArray
//    val tensor = Tensor(data, shape)
//    getParametersFromModel(model)._1.copy(tensor)
//  }
//
//  def getTensor(name: String, modelData: Table): Tensor[Float] = {
//    val dataMap = modelData.getTableMap.get(name)
//    val data = dataMap.getTensorList.asScala.map(Float2float).toArray
//    val shape = dataMap.getShapeList.asScala.map(Integer2int).toArray
//    Tensor[Float](data, shape)
//  }
//
//
//  def downloadModel(client: ClientGrpc, modelName: String, flVersion: Int): ParameterServerServiceInterface.Table = {
//    var code = 0
//    var maxRetry = 20000
//    var downloadRes = None: Option[ParameterServerServiceInterface.DownloadResponse]
//    while (code == 0 && maxRetry > 0) {
//      downloadRes = Some(client.downloadData(modelName, flVersion))
//      code = downloadRes.get.getCode
//      if (code == 0) {
//        logger.info("Waiting 10ms for other clients!")
//        maxRetry -= 1
//        Thread.sleep(10)
//      }
//    }
//    if (code == 0) throw new Exception("Failed to download model within max retries!")
//    downloadRes.get.getData
//  }
//
//  def randomSplit[T: ClassTag](weight: Array[Float],
//                               data: Array[T],
//                               seed: Int = 1): Array[Array[T]] = {
//    val random = new Random(seed = seed)
//    val lens = weight.map(v => (v * data.length).toInt)
//    lens(lens.length - 1) = data.length - lens.slice(0, lens.length - 1).sum
//    val splits = lens.map(len => new Array[T](len))
//    val counts = lens.map(_ => 0)
//    data.foreach{d =>
//      var indx = random.nextInt(weight.length)
//      while(counts(indx) == lens(indx)){
//        indx = (indx + 1) % weight.length
//      }
//      splits(indx)(counts(indx)) = d
//      counts(indx) += 1
//    }
//    splits
//  }
//
//  def toFloatTensor(t: Tensor[Float]): FloatTensor = {
//    FloatTensor
//      .newBuilder()
//      .addAllTensor(t.toTensor[Float].contiguous().storage()
//        .array().slice(t.storageOffset() - 1, t.storageOffset() - 1 + t.nElement())
//        .map(float2Float).toIterable.asJava)
//      .addAllShape(t.toTensor[Float].size()
//        .map(int2Integer).toIterable.asJava)
//      .build()
//  }
//
//  def almostEqual(v1: Float, v2: Float): Boolean = {
//    if (math.abs(v1 - v2) <= 1e-1f)
//      true
//    else
//      false
//  }
//
//
//
//  def uploadOutput(client: ClientGrpc, model: Module[Float], flVersion: Int, target: Activity = null): Unit = {
//    val metadata = TableMetaData.newBuilder
//      .setName(s"${model.getName()}_output").setVersion(flVersion).build
//
//    // TODO: support table output and table target
//    val output = model.output.toTensor[Float]
//
//    val tensor = toFloatTensor(output)
//
//    val modelData = Table.newBuilder
//      .putTable("output", tensor)
//      .setMetaData(metadata)
//    if (target != null) {
//      val targetTensor = toFloatTensor(target.toTensor[Float])
//      modelData.putTable("target", targetTensor)
//    }
//    client.uploadData(modelData.build())
//  }
//
//  def evaluateOutput(
//      client: ClientGrpc,
//      model: Module[Float],
//      flVersion: Int,
//      target: Activity,
//      lastBatch: Boolean): EvaluateResponse = {
//    val metadata = TableMetaData.newBuilder
//      .setName(s"${model.getName()}_output").setVersion(flVersion).build
//
//    // TODO: support table output and table target
//    val output = model.output.toTensor[Float]
//
//    val tensor = toFloatTensor(output)
//
//    val modelData = Table.newBuilder
//      .putTable("output", tensor)
//      .setMetaData(metadata)
//    if (target != null) {
//      val targetTensor = toFloatTensor(target.toTensor[Float])
//      modelData.putTable("target", targetTensor)
//    }
//    client.evaluate(modelData.build(), lastBatch)
//  }
}
