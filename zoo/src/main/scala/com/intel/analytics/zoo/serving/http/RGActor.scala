package com.intel.analytics.zoo.serving.http

import java.util

import akka.actor.ActorRef
import com.intel.analytics.zoo.serving.arrow.ArrowDeserializer
import com.intel.analytics.zoo.serving.utils.Conventions
import org.slf4j.LoggerFactory

import scala.collection.JavaConverters._
import scala.collection.mutable

class RGActor(
                     redisHost: String,
                     redisPort: Int,
                     redisInputQueue: String,
                     redisOutputQueue: String,
                     redisSecureEnabled: Boolean,
                     redissTrustStorePath: String,
                     redissTrustStoreToken: String) extends JedisEnabledActor {
  override val logger = LoggerFactory.getLogger(classOf[RedisGetActor])
  val jedis = retrieveJedis(redisHost, redisPort,
    redisSecureEnabled, redissTrustStorePath, redissTrustStoreToken)
  var requestMap = mutable.Map[String, ActorRef]()

  override def receive: Receive = {
    case message: PutEndMessage =>
      logger.info(s"PutEndMessage received from ${sender().path.name} at ${System.currentTimeMillis()}")
      requestMap += (message.actor.path.name -> message.actor)
    case message: DequeueMessage => {
      logger.info(s"DequeueMessage received at ${System.currentTimeMillis()}")
      getAll(redisOutputQueue).foreach(result => {
        val queryOption = requestMap.get(result._1).get
        if (queryOption != null) {
          val queryActor = requestMap.get(result._1).asInstanceOf[ActorRef]
          val queryResult = result._2.asScala
          println(queryResult.toString())
          queryActor ! ModelOutputMessage(queryResult)
        }

      })
    }
  }

  def get(queue: String, ids: Seq[String]): Seq[(String, util.Map[String, String])] = {
    silent(s"$actorName get response from redis")(FrontEndApp.getRedisTimer) {
      ids.map(id => {
        val key = s"$queue$id"
        (id, jedis.hgetAll(key))
      }).filter(!_._2.isEmpty)
    }
  }
  def getAll(queue: String): mutable.Set[(String, util.Map[String, String])] = {
    val resultSet = jedis.keys(s"${queue}*")
    val res = resultSet.asScala.map(key => {
      (key, jedis.hgetAll(key))

    })
    resultSet.asScala.foreach(key => jedis.del(key))
    res
  }

}