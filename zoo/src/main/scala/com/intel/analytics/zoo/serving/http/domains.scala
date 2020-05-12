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

package com.intel.analytics.zoo.serving.http

import java.util.{HashMap, UUID}

import akka.actor.ActorRef
import com.codahale.metrics.Timer

sealed trait ServingMessage
case class PredictionInputMessage(input: PredictionInput) extends ServingMessage
case class PredictionInputFlushMessage() extends ServingMessage
case class PredictionQueryMessage(id: String) extends ServingMessage
case class PredictionQueryWithTargetMessage(query: PredictionQueryMessage, target: ActorRef)
  extends ServingMessage

sealed trait PredictionInput {
  def getId(): String
  def toHash(): HashMap[String, String]
}
case class BytesPredictionInput(uuid: String, bytesStr: String) extends PredictionInput {
  override def getId(): String = this.uuid
  def toMap(): Map[String, String] = Map("uuid" -> uuid, "bytesStr" -> bytesStr)
  override def toHash(): HashMap[String, String] = {
    val hash = new HashMap[String, String]()
    hash.put("uri", uuid)
    hash.put("image", bytesStr)
    hash
  }
}
object BytesPredictionInput {
  def apply(str: String): BytesPredictionInput =
    BytesPredictionInput(UUID.randomUUID().toString, str)
}

case class PredictionOutput[Type](uuid: String, result: Type)

class ImageFeature(val b64: String)

case class Instances(instances: List[Map[String, Any]])
object Instances{
  def apply(instances: Map[String, Any]*): Instances = {
    Instances(instances.toList)
  }
}

case class Predictions[Type](predictions: Array[Type]) {
  override def toString: String = JsonUtil.toJson(this)
}
object Predictions {
  def apply[T](output: PredictionOutput[T])(implicit m: Manifest[T]): Predictions[T] = {
    Predictions(Array(output.result))
  }
  def apply[T](outputs: List[PredictionOutput[T]])(implicit m: Manifest[T]): Predictions[T] = {
    Predictions(outputs.map(_.result).toArray)
  }
}


case class ServingResponse[Type](statusCode: Int, entity: Type) {
  def this(tuple: (Int, Type)) = this(tuple._1, tuple._2)
  override def toString: String = s"[$statusCode, $entity]"
  def isSuccessful: Boolean = statusCode / 100 == 2
}

case class ServingRuntimeException(message: String = null, cause: Throwable = null)
  extends RuntimeException(message, cause) {
  def this(response: ServingResponse[String]) = this(JsonUtil.toJson(response), null)
}

case class ServingError(error: String) {
  override def toString: String = JsonUtil.toJson(this)
}

case class ServingTimerMetrics(
    name: String,
    count: Long,
    meanRate: Double,
    min: Long,
    max: Long,
    mean: Double,
    median: Double,
    stdDev: Double,
    _75thPercentile: Double,
    _95thPercentile: Double,
    _98thPercentile: Double,
    _99thPercentile: Double,
    _999thPercentile: Double
)

object ServingTimerMetrics {
  def apply(name: String, timer: Timer): ServingTimerMetrics =
    ServingTimerMetrics(
      name,
      timer.getCount,
      timer.getMeanRate,
      timer.getSnapshot.getMin/1000000,
      timer.getSnapshot.getMax/1000000,
      timer.getSnapshot.getMean/1000000,
      timer.getSnapshot.getMedian/1000000,
      timer.getSnapshot.getStdDev/1000000,
      timer.getSnapshot.get75thPercentile()/1000000,
      timer.getSnapshot.get95thPercentile()/1000000,
      timer.getSnapshot.get98thPercentile()/1000000,
      timer.getSnapshot.get99thPercentile()/1000000,
      timer.getSnapshot.get999thPercentile()/1000000
    )
}
