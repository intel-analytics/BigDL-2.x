package com.intel.analytics.zoo.serving.http

import java.util

import akka.actor.{Actor, ActorRef}
import com.intel.analytics.zoo.serving.serialization.ArrowDeserializer
import com.intel.analytics.zoo.serving.pipeline.{RedisIO, RedisUtils}
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
    RedisIO.getRedisClient(new JedisPool())
  } else {
    RedisIO.getRedisClient(jedisPool)
  }
  var requestMap = mutable.Map[String, ActorRef]()

  override def receive: Receive = {
    case message: DataInputMessage =>
      timing(s"${self.path.name} input message process")() {
        val predictionInput = message.inputs.head
        logger.info(s"${System.currentTimeMillis()} Input enqueue $predictionInput at time ")
        enqueue(redisInputQueue, predictionInput)

        requestMap += (Conventions.RESULT_PREFIX + Conventions.SERVING_STREAM_DEFAULT_NAME + ":" + predictionInput.getId() -> sender())
      }
    case message: DequeueMessage => {
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
  def enqueue(queue: String, input: PredictionInput): Unit = {
    timing(s"${self.path.name} put request to redis")(FrontEndApp.putRedisTimer) {
      val hash = input.toHashByStream()
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