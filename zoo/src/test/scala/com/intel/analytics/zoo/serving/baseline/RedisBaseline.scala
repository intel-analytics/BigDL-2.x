package com.intel.analytics.zoo.serving.baseline

import java.io.{ByteArrayOutputStream, ObjectOutputStream}
import java.util.HashMap

import com.intel.analytics.zoo.serving.http.{Instances, InstancesPredictionInput, JsonUtil}
import com.intel.analytics.zoo.serving.pipeline.RedisUtils
import com.intel.analytics.zoo.serving.utils.Supportive
import org.scalatest.{FlatSpec, Matchers}
import redis.clients.jedis.Jedis

class RedisBaseline extends FlatSpec with Matchers with Supportive {

  "redis put" should "work" in {
    val jedis = new Jedis()
    RedisUtils.createRedisGroupIfNotExist(jedis, "tmp_stream")
    val resource = getClass().getClassLoader().getResource("serving")
    val dataPath = resource.getPath + "/ndarray-128-base64"
    val b64string = scala.io.Source.fromFile(dataPath).mkString
    val hash = new HashMap[String, String]()
    hash.put("key", b64string)
    timing("put"){
      (0 to 100).foreach(_ => jedis.xadd("tmp_stream", null, hash))

    }

    jedis.xgroupDestroy("tmp_stream", "serving")
  }
  "dien client to redis" should "work" in {
    val jedis = new Jedis()
    RedisUtils.createRedisGroupIfNotExist(jedis, "tmp_stream")
    val resource = getClass().getClassLoader().getResource("serving")
    val dataPath = resource.getPath + "/dien_json_str.json"
    val jsonString = scala.io.Source.fromFile(dataPath).mkString
    val instances = timing("json encode to Instance") {
      JsonUtil.fromJson(classOf[Instances], jsonString)
    }
    val inputs = instances.instances.map(instance => {
      InstancesPredictionInput(Instances(instance))
    })
    inputs.foreach(input => {
      timing(s"put request to redis") {
        for ( i <- 0 to 100) {
          timing("total one record in to redis") {
            val hash = timing("to hash") {
              val hash = new HashMap[String, String]()
              val bytes = timing("instance to arrow") {
                input.instances.toArrow()
              }

              val b64 = java.util.Base64.getEncoder.encodeToString(bytes)
              hash.put("uri", input.uuid)
              hash.put("data", b64)
              hash
            }
            timing("put one"){
              jedis.xadd("tmp_stream", null, hash)
            }
          }

        }
      }
    })
  }
  "dien client to redis using java serialization" should "work" in {
    val jedis = new Jedis()
    RedisUtils.createRedisGroupIfNotExist(jedis, "tmp_stream")
    val resource = getClass().getClassLoader().getResource("serving")
    val dataPath = resource.getPath + "/dien_json_str.json"
    val jsonString = scala.io.Source.fromFile(dataPath).mkString
    val instances = timing("json encode to Instance") {
      JsonUtil.fromJson(classOf[Instances], jsonString)
    }
    val inputs = instances.instances.map(instance => {
      InstancesPredictionInput(Instances(instance))
    })
    inputs.foreach(input => {
      timing(s"put request to redis") {
        for ( i <- 0 to 100) {
          timing("total one record in to redis") {
            val hash = timing("to hash") {
              val hash = new HashMap[String, String]()
              val bytes = timing("instance to byte array") {
                val bos = new ByteArrayOutputStream()
                val out = new ObjectOutputStream(bos)
                out.writeObject(input.instances)
                out.flush()
                bos.toByteArray()
              }

              val b64 = java.util.Base64.getEncoder.encodeToString(bytes)
              hash.put("uri", input.uuid)
              hash.put("data", b64)
              hash
            }
            timing("put one"){
              jedis.xadd("tmp_stream", null, hash)
            }
          }

        }
      }
    })
  }
}
