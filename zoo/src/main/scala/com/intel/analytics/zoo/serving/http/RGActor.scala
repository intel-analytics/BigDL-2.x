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
  val requestMap = Map[String, ActorRef]()

  override def receive: Receive = {
    case message: PredictionQueryMessage =>
      val results = get(redisOutputQueue, message.ids)
      if (null != results && results.size == message.ids.size) {
        results.foreach(x => {
          val b64string = x._2.get("value")
          try {
            x._2.put("value", ArrowDeserializer(b64string))
          } catch {
            case _: Exception =>
          }
        })

        sender() ! results
        // result get, remove in redis here
        message.ids.foreach(id =>
          jedis.del(Conventions.RESULT_PREFIX + Conventions.SERVING_STREAM_DEFAULT_NAME + ":" + id)
        )
      } else {
        sender() ! Seq[(String, util.Map[String, String])]()
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
  def getAll(queue: String, ids: Seq[String]): mutable.Set[(String, util.Map[String, String])] = {
    val resultSet = jedis.keys(s"${queue}:*")
    resultSet.asScala.map(key => {
      (key, jedis.hgetAll(key))
    })
  }

}