package com.intel.analytics.zoo.serving.http

import akka.actor.ActorRef

import scala.collection.mutable

class AbstractActor {

}
case class PutEndMessage(key: String, actor: ActorRef)
case class DequeueMessage()
case class ModelOutputMessage(valueMap: mutable.Map[String, String])
case class DataInputMessage(inputs: Seq[PredictionInput])

case class TestInputMessage(inputs: String)
case class TestOutputMessage(inputs: String)