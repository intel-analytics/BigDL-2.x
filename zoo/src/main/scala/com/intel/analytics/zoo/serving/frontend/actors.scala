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

package com.intel.analytics.zoo.serving.frontend

import akka.actor.{Actor, ActorRef}
import org.slf4j.LoggerFactory

class RedisPutActor(
    redisHost: String,
    redisPort: Int,
    redisInputQueue: String,
    redisOutputQueue: String) extends Actor with Supportive {
  override val logger = LoggerFactory.getLogger(classOf[RedisPutActor])
  val actorName = self.path.name

  override def receive: Receive = {
    case _ => ""
  }
}

class RedisGetActor(
    redisHost: String,
    redisPort: Int,
    redisInputQueue: String,
    redisOutputQueue: String) extends Actor with Supportive {
  override val logger = LoggerFactory.getLogger(classOf[RedisPutActor])
  val actorName = self.path.name

  override def receive: Receive = {
    case _ => ""
  }
}

class QueryActor(redisGetActor: ActorRef) extends Actor with Supportive {
  override val logger = LoggerFactory.getLogger(classOf[RedisPutActor])
  val actorName = self.path.name

  override def receive: Receive = {
    case _ => ""
  }
}