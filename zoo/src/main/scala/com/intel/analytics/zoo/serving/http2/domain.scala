package com.intel.analytics.zoo.serving.http2

import java.util
import java.util.concurrent.LinkedBlockingQueue

import akka.actor.{ActorRef, Props}
import akka.pattern.ask
import com.fasterxml.jackson.annotation.JsonSubTypes.Type
import com.fasterxml.jackson.annotation.{JsonSubTypes, JsonTypeInfo}
import com.fasterxml.jackson.databind.ObjectMapper
import com.intel.analytics.zoo.pipeline.inference.InferenceModel
import com.intel.analytics.zoo.serving.http.FrontEndApp.{overallRequestTimer, silent, system, timeout, timing, waitRedisTimer}
import com.intel.analytics.zoo.serving.http._

import scala.collection.mutable
import scala.concurrent.Await

class ServableManager {
  private var modelVersionMap = new mutable.HashMap[String, mutable.HashMap[String, Servable]]

  def load(dir: String) = {

  }

  def retriveModels(modelName: String): List[Servable] = {
    if (!modelVersionMap.contains(modelName)) {
      throw ServingRuntimeException("model not exist", null)
    }
    modelVersionMap(modelName).values.toList
  }

  def retriveModel(modelName: String, modelVersion: String): Servable = {
    if (!modelVersionMap.contains(modelName) || !modelVersionMap(modelName).contains(modelVersion)) {
      throw ServingRuntimeException("model not exist", null)
    }
    modelVersionMap(modelName)(modelVersion)
  }
}

abstract class Servable(modelMetaData: ModelMetaData) {
  def predict(inputs: Seq[PredictionInput]):
  Seq[PredictionOutput[String]]

  def load(): Unit
}

class InferenceModelServable(inferenceModelMetaData: InferenceModelMetaData) extends Servable(inferenceModelMetaData) {
  var model: InferenceModel = _

  def load(): Unit = {
    model = new InferenceModel(1)
    model.doLoadOpenVINO(inferenceModelMetaData.modelPath,
      inferenceModelMetaData.weightPath)
  }

  def predict(inputs: Seq[PredictionInput]):
  Seq[PredictionOutput[String]]  = {
    null
  }
}

class ClusterServingServable(clusterServingMetaData: ClusterServingMetaData) extends Servable(clusterServingMetaData) {
  var redisPutter: ActorRef = _

  val redisGetterName = s"redis-getter"
  val redisGetter = timing(s"$redisGetterName initialized.")() {
    val redisGetterProps = Props(new RedisPutActor(
      clusterServingMetaData.redisHost, clusterServingMetaData.redisPort.toInt,
      clusterServingMetaData.redisInputQueue, clusterServingMetaData.redisOutputQueue,
      0, 56, false, null, "1234qwer"))
    system.actorOf(redisGetterProps, name = redisGetterName)
  }

  val querierNum = 1000
  val querierQueue = timing(s"queriers initialized.")() {
    val querierQueue = new LinkedBlockingQueue[ActorRef](querierNum)
    val querierProps = Props(new QueryActor(redisGetter))
    List.range(0, querierNum).map(index => {
      val querierName = s"querier-$index"
      val querier = system.actorOf(querierProps, name = querierName)
      querierQueue.put(querier)
    })
    querierQueue
  }

  def load(): Unit = {
    val redisPutterName = s"redis-putter"
    redisPutter = timing(s"$redisPutterName initialized.")() {
      val redisPutterProps = Props(new RedisPutActor(
        clusterServingMetaData.redisHost,
        clusterServingMetaData.redisPort.toInt,
        clusterServingMetaData.redisInputQueue,
        clusterServingMetaData.redisOutputQueue,
        0, //TODO:
        56,
        false,
        null,
        "1234qwer"))
      system.actorOf(redisPutterProps, name = redisPutterName)
    }

  }

  def predict(inputs: Seq[PredictionInput]):
  Seq[PredictionOutput[String]] = {
    silent("put message send")() {
      val message = PredictionInputMessage(inputs)
      redisPutter ! message
    }
    val result = silent("response waiting")() {
      val ids = inputs.map(_.getId())
      val queryMessage = PredictionQueryMessage(ids)
      val querier = silent("querier take")() {
        querierQueue.take()
      }
      val results = timing(s"query message wait for key $ids")(
        overallRequestTimer, waitRedisTimer) {
        Await.result(querier ? queryMessage, timeout.duration)
          .asInstanceOf[Seq[(String, util.Map[String, String])]]
      }
      silent("querier back")() {
        querierQueue.offer(querier)
      }
      val objectMapper = new ObjectMapper()
      results.map(r => {
        val resultStr = objectMapper.writeValueAsString(r._2)
        PredictionOutput(r._1, resultStr)
      })
    }
    result
  }

}


@JsonTypeInfo(
  use = JsonTypeInfo.Id.NAME,
  include = JsonTypeInfo.As.PROPERTY,
  property = "type"
)
@JsonSubTypes(Array(
  new Type(value = classOf[InferenceModelMetaData], name = "InferenceModelMetaData"),
  new Type(value = classOf[ClusterServingMetaData], name = "ClusterServingMetaData")
))
abstract class ModelMetaData(modelName: String, modelVersion: String) {

}


case class InferenceModelMetaData(modelName: String,
                                  modelVersion: String,
                                  modelPath: String,
                                  weightPath: String,
                                  modelType: String)
  extends ModelMetaData(modelName, modelVersion)

case class ClusterServingMetaData(modelName: String,
                                  modelVersion: String,
                                  redisHost: String,
                                  redisPort: String,
                                  redisInputQueue: String,
                                  redisOutputQueue: String) extends ModelMetaData(modelName, modelVersion)


case class ServableManagerConf(modelMetaDataList: List[ModelMetaData])

case class ServingRuntimeException(message: String = null, cause: Throwable = null)
  extends RuntimeException(message, cause) {
  def this(response: ServingResponse[String]) = this(JsonUtil.toJson(response), null)
}