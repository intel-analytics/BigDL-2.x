package com.intel.analytics.zoo.serving

import org.apache.log4j.Logger
import org.scalatest.{FlatSpec, Matchers}

import sys.process._
import redis.clients.jedis.Jedis
import redis.embedded.RedisServer

import scala.io.Source
import scala.collection.JavaConverters._
import scala.concurrent.Future
import scala.concurrent.ExecutionContext.Implicits.global

class CorrectnessSpec extends FlatSpec with Matchers {
  val configPath = "/tmp/config.yaml"
  val logger = Logger.getLogger(getClass)
  def runServingBg(): Future[Unit] = Future {
    ClusterServing.run(configPath)
  }
  "Cluster Serving result" should "be correct" in {
    val redisServer = new RedisServer()
    redisServer.start()
    val cli = new Jedis()

    // call push method in python
    "wget -O /tmp/serving_val.tar http://10.239.45.10:8081/repository/raw/analytics-zoo-data/imagenet_1k.tar".!
    "tar -xvf /tmp/serving_val.tar -C /tmp/".!
    runServingBg().onComplete(_ => None)

    val imagePath = "/tmp/imagenet_1k"
    val lsCmd = "ls " + imagePath

    val totalNum = (lsCmd #| "wc").!!.split(" +").filter(_ != "").head.toInt


    val enqueueScriptPathCmd = "python3 " + getClass.getClassLoader.getResource("serving/enqueue_image_in_path.py").getPath +
      " --img_path " + imagePath + " --img_num " + totalNum.toString
    val p = Process(enqueueScriptPathCmd, None
      ,
      "PYTHONPATH" -> "$PYTHONPATH:/home/litchy/pro/analytics-zoo/dist/lib/analytics-zoo-bigdl_0.10.0-spark_2.4.3-0.8.0-SNAPSHOT-python-api.zip",
    "SPARK_HOME" -> "/home/litchy/Programs/spark-2.4.0-bin-hadoop2.7"
    )
    p.!
    ("rm -rf /tmp/" + imagePath + "*").!
    "rm -rf /tmp/serving_val_*".!
    "rm -rf /tmp/config.yaml".!
    // check if record is enough
    var cnt = 0
    var res_length: Int = 0
    while (res_length != totalNum) {
      val res_list = cli.keys("result:*")
      res_length = res_list.size()
      Thread.sleep(5000)
      cnt += 1
      if (cnt >= 100) {
        throw new Error("validation fails, data maybe lost")
      }
      logger.info(s"Current records in Redis:${res_length}")

    }
    // record enough start validation,
    // generate key first
    var top1_dict = Map[String, String]()
    val res_list = cli.keys("result:*")
    res_list.asScala.foreach(key => {
      val res = cli.hgetAll(key).get("value")

      val cls = res.substring(2, res.length - 2).split(",").head
      top1_dict += (key.stripPrefix("result:") -> cls)
      top1_dict
    })
    // start check with txt file
    redisServer.stop()
    logger.info("Redis server stopped")
    var cN = 0f
    var tN = 0f
    for (line <- Source.fromFile(imagePath + ".txt").getLines()) {
      val key = line.split(" ").head
      val cls = line.split(" ").tail(0)
      if (top1_dict(key) == cls) {
        cN += 1
      }
      tN += 1
    }
    val acc = cN / tN
    logger.info(s"Top 1 Accuracy of serving, Openvino ResNet50 Model on ImageNet is ${acc}")
    assert(acc > 0.72)

  }



}
