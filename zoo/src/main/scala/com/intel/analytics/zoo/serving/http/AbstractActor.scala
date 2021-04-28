package com.intel.analytics.zoo.serving.http

import akka.actor.ActorRef
import com.intel.analytics.bigdl.nn.abstractnn.Activity

import scala.collection.mutable

class AbstractActor {

}
case class DequeueMessage()
case class ModelOutputMessage(valueMap: mutable.Map[String, String])
case class DataInputMessage(id: String, inputs: String)
