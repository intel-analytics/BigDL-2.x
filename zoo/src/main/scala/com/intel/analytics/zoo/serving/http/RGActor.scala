package com.intel.analytics.zoo.serving.http

import java.util

import akka.actor.ActorRef
import com.intel.analytics.zoo.serving.arrow.ArrowDeserializer
import com.intel.analytics.zoo.serving.utils.Conventions
import org.slf4j.LoggerFactory

import scala.collection.JavaConverters._
import scala.collection.mutable

class RGActor(redisOutputQueue: String = Conventions.RESULT_PREFIX + Conventions.SERVING_STREAM_DEFAULT_NAME + ":"
                     ) extends JedisEnabledActor {
  override val logger = LoggerFactory.getLogger(getClass)
  val jedis = retrieveJedis()
  var requestMap = mutable.Map[String, ActorRef]()

  override def receive: Receive = {
    case message: PutEndMessage =>
      logger.info(s"${System.currentTimeMillis()} PutEndMessage received from ${sender().path.name} at time")
      requestMap += (Conventions.RESULT_PREFIX + Conventions.SERVING_STREAM_DEFAULT_NAME + ":" + message.key -> sender())
      logger.info(s"result map is currently $requestMap")
    case message: DequeueMessage => {
//      logger.info(s"${System.currentTimeMillis()} Dequeue at time ")
      getAll(redisOutputQueue).foreach(result => {
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