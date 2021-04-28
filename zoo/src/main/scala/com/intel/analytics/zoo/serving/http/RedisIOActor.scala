package com.intel.analytics.zoo.serving.http

import java.util
import java.util.{HashMap, UUID}

import akka.actor.{Actor, ActorRef}
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.zoo.serving.serialization.{ArrowDeserializer, StreamSerializer}
import com.intel.analytics.zoo.serving.pipeline.RedisUtils
import com.intel.analytics.zoo.serving.utils.Conventions
import org.slf4j.LoggerFactory
import redis.clients.jedis.JedisPool

import scala.collection.JavaConverters._
import scala.collection.mutable

class RedisIOActor(redisOutputQueue: String = Conventions.RESULT_PREFIX + Conventions.SERVING_STREAM_DEFAULT_NAME + ":",
                   redisInputQueue: String = "serving_stream",
                   jedisPool: JedisPool = null) extends Actor with Supportive {
  override val logger = LoggerFactory.getLogger(getClass)
  val jedis = if (jedisPool == null) {
    RedisUtils.getRedisClient(new JedisPool())
  } else {
    RedisUtils.getRedisClient(jedisPool)
  }
  var requestMap = mutable.Map[String, ActorRef]()

  override def receive: Receive = {
    case message: DataInputMessage =>
      silent(s"${self.path.name} input message process")() {
        logger.info(s"${System.currentTimeMillis()} Input enqueue ${message.id} at time ")
        enqueue(redisInputQueue, message)

        requestMap += (Conventions.RESULT_PREFIX + Conventions.SERVING_STREAM_DEFAULT_NAME + ":" + message.id -> sender())
      }
    case message: DequeueMessage => {
        if (!requestMap.isEmpty) {
          dequeue(redisOutputQueue).foreach(result => {
            logger.info(s"${System.currentTimeMillis()} Get redis result at time ")
            val queryOption = requestMap.get(result._1)
            if (queryOption != None) {
              val queryResult = result._2.asScala
              queryOption.get ! ModelOutputMessage(queryResult)
              requestMap -= result._1
              logger.info(s"${System.currentTimeMillis()} Send ${result._1} back at time ")
            }

          })
        }
      }
  }
  def enqueue(queue: String, input: DataInputMessage): Unit = {
    timing(s"${self.path.name} put request to redis")(FrontEndApp.putRedisTimer) {
      val hash = new HashMap[String, String]()
//      val bytes = StreamSerializer.objToBytes(input.inputs)
//      val b64 = java.util.Base64.getEncoder.encodeToString(bytes)
      hash.put("uri", input.id)
      hash.put("data", input.inputs)
      hash.put("serde", "stream")
      jedis.xadd(queue, null, hash)
    }
  }
  def dequeue(queue: String): mutable.Set[(String, util.Map[String, String])] = {
    val resultSet = jedis.keys(s"${queue}*")
    val res = resultSet.asScala.map(key => {
      (key, jedis.hgetAll(key))

    })
    resultSet.asScala.foreach(key => jedis.del(key))
    res
  }

}