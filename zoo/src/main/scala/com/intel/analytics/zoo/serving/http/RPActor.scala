package com.intel.analytics.zoo.serving.http

import java.util
import java.util.concurrent.TimeUnit
import akka.pattern.ask
import akka.actor.ActorRef
import akka.util.Timeout
import com.intel.analytics.zoo.serving.utils.Conventions
import org.slf4j.LoggerFactory

import scala.collection.mutable
import scala.concurrent.Await

class RPActor(
                     redisHost: String,
                     redisPort: Int,
                     redisInputQueue: String,
                     redisOutputQueue: String,
                     timeWindow: Int,
                     countWindow: Int,
                     redisSecureEnabled: Boolean,
                     redissTrustStorePath: String,
                     redissTrustStoreToken: String,
                     redisGetActor: ActorRef) extends JedisEnabledActor {
  override val logger = LoggerFactory.getLogger(classOf[RedisPutActor])
  val jedis = retrieveJedis(redisHost, redisPort,
    redisSecureEnabled, redissTrustStorePath, redissTrustStoreToken)
  implicit val timeout: Timeout = Timeout(100, TimeUnit.SECONDS)
  var start = System.currentTimeMillis()
  val cache = Set[PredictionInput]()

  override def receive: Receive = {
    case message: DataInputMessage =>
      silent(s"$actorName input message process, ${cache.size}")() {
        val predictionInputs = message.inputs
        predictionInputs.foreach(x => {
          put(redisInputQueue, x)
          logger.info(s"Input enqueue $x at time ${System.currentTimeMillis()}")
        })
      }
      logger.info(s"sending put end message to ${redisGetActor.path.name}")
      val res = Await.result(redisGetActor ? PutEndMessage, timeout.duration)
      sender() ! res

    case message: SecuredModelSecretSaltMessage =>
      silent(s"$actorName put secret and salt in redis")() {
        jedis.hset(Conventions.MODEL_SECURED_KEY, Conventions.MODEL_SECURED_SECRET, message.secret)
        jedis.hset(Conventions.MODEL_SECURED_KEY, Conventions.MODEL_SECURED_SALT, message.salt)
        sender() ! true
      }
  }

  def put(queue: String, input: PredictionInput): Unit = {
    timing(s"$actorName put request to redis")(FrontEndApp.putRedisTimer) {
      val hash = input.toHash()
      jedis.xadd(queue, null, hash)
    }
  }

  def putInPipeline(queue: String, inputs: mutable.Set[PredictionInput]): Unit = {
    average(s"$actorName put ${inputs.size} requests to redis")(inputs.size)(
      FrontEndApp.putRedisTimer) {
      val pipeline = jedis.pipelined()
      inputs.map(input => {
        val hash = input.toHash()
        pipeline.xadd(queue, null, hash)
      })
      pipeline.sync()
      inputs.clear()
    }
  }

  def putInTransaction(queue: String, inputs: mutable.Set[PredictionInput]): Unit = {
    average(s"$actorName put ${inputs.size} requests to redis")(inputs.size)(
      FrontEndApp.putRedisTimer) {
      val t = jedis.multi();
      inputs.map(input => {
        val hash = input.toHash()
        t.xadd(queue, null, hash)
        println("put input", System.currentTimeMillis(), input)
      })
      t.exec()
      logger.info(s"${System.currentTimeMillis}, ${inputs.map(_.getId).mkString(",")}")
      inputs.clear()
    }
  }
}
