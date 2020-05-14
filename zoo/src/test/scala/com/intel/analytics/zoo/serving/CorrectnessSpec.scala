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

package com.intel.analytics.zoo.serving

import java.io.{ByteArrayOutputStream, File, PrintWriter}
import java.util.Base64

import org.apache.log4j.Logger
import org.scalatest.{FlatSpec, Matchers}

import sys.process._
import redis.clients.jedis.{Jedis, StreamEntryID}

import scala.io.Source
import scala.collection.JavaConverters._
import scala.concurrent.Future
import scala.concurrent.ExecutionContext.Implicits.global
import com.intel.analytics.bigdl.opencv.OpenCV
import javax.imageio.ImageIO
import java.awt._
import java.awt.image.BufferedImage

import scala.util.Random
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import org.opencv.core._
import org.apache.commons.io.FileUtils
import com.intel.analytics.zoo.feature.image._

class CorrectnessSpec extends FlatSpec with Matchers {
  val configPath = "/tmp/config.yaml"
//  val configPath = "/home/litchy/pro/analytics-zoo/config.yaml"
  var redisHost: String = "localhost"
  var redisPort: Int = 6379
  val logger = Logger.getLogger(getClass)
  def resize(p: String): String = {
    val source = ImageIO.read(new File(p))
    val outputImage: BufferedImage = new BufferedImage(224, 224, source.getType)
    val graphics2D: Graphics2D = outputImage.createGraphics

    graphics2D.drawImage(source, 0, 0, 224, 224, null)
    graphics2D.dispose()

    val byteStream = new ByteArrayOutputStream()
    ImageIO.write(outputImage, "jpg", byteStream)

//    val f = new File("/home/litchy/tmp/034.jpg")
//    ImageIO.write(outputImage, "jpg", f)
    val dataStr = Base64.getEncoder.encodeToString(byteStream.toByteArray)
    dataStr
  }
  def getBase64FromPath(path: String): String = {

//    val img = Imgcodecs.imread(path)
//
    val b = FileUtils.readFileToByteArray(new File(path))
    val img = OpenCVMethod.fromImageBytes(b, Imgcodecs.CV_LOAD_IMAGE_COLOR)
    Imgproc.resize(img, img, new Size(224, 224))
    val matOfByte = new MatOfByte()
    Imgcodecs.imencode(".jpg", img, matOfByte)
    val dataStr = Base64.getEncoder.encodeToString(matOfByte.toArray)
    dataStr
  }


  def runServingBg(): Future[Unit] = Future {
    ClusterServing.run(configPath, redisHost, redisPort)
  }
  "Cluster Serving result" should "be correct" in {
    redisHost = "10.239.47.210"
    redisPort = 16380
    val cli = new Jedis(redisHost, redisPort)

    cli.flushAll()

    cli.xgroupCreate("image_stream", "serving",
      new StreamEntryID(0, 0), true)
    Thread.sleep(3000)


    ("wget -O /tmp/serving_val.tar http://10.239.45.10:8081" +
      "/repository/raw/analytics-zoo-data/imagenet_1k.tar").!
    "tar -xvf /tmp/serving_val.tar -C /tmp/".!
    runServingBg().onComplete(_ => None)
    Thread.sleep(10000)
    val imagePath = "/tmp/imagenet_1k"
//    val imagePath = "/home/litchy/tmp/imagenet_1k"
    val lsCmd = "ls " + imagePath

    val totalNum = (lsCmd #| "wc").!!.split(" +").filter(_ != "").head.toInt

    // enqueue image
    val f = new File(imagePath)
    val fileList = f.listFiles
    logger.info(s"${fileList.size} images about to enqueue...")

    for (file <- fileList) {
      val dataStr = getBase64FromPath(file.getAbsolutePath)
      val infoMap = Map[String, String]("uri" -> file.getName, "image" -> dataStr)
      cli.xadd("image_stream", StreamEntryID.NEW_ENTRY, infoMap.asJava)
    }

    //    val enqueueScriptPathCmd = "python3 " +
    //      getClass.getClassLoader.getResource("serving/enqueue_image_in_path.py").getPath +
    //      " --img_path " + imagePath + " --img_num " +
    //      totalNum.toString + " --host 172.168.2.102 --port 6379"
    //    val p = Process(enqueueScriptPathCmd, None
    //      ,
    //      "PYTHONPATH" -> "$PYTHONPATH:/home/litchy/pro/analytics-zoo/dist/lib/
    //            analytics-zoo-bigdl_0.10.0-spark_2.4.3-0.8.0-SNAPSHOT-python-api.zip",
    //    "SPARK_HOME" -> "/home/litchy/Programs/spark-2.4.0-bin-hadoop2.7"
    //    )
    //    p.!
    ("rm -rf /tmp/" + imagePath + "*").!
    "rm -rf /tmp/serving_val_*".!
    "rm -rf /tmp/config.yaml".!
    // check if record is enough
    var cnt: Int = 0
    var res_length: Int = 0
    while (res_length != totalNum) {
      val res_list = cli.keys("result:*")
      res_length = res_list.size()
      Thread.sleep(10000)
      cnt += 1
      if (cnt >= 150 || (cnt >= 25 && res_length == 0)) {
        logger.info(s"count is ${cnt}")
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

    logger.info("Redis server stopped")
    var cN = 0f
    var tN = 0f
    for (line <- Source.fromFile(imagePath + ".txt").getLines()) {
      val key = line.split(" ").head
      val cls = line.split(" ").tail(0)
      try {
        if (top1_dict(key) == cls) {
          cN += 1
        }
        tN += 1
      }
      catch {
        case _ => None
      }
    }
    val acc = cN / tN
    logger.info(s"Top 1 Accuracy of serving, Openvino ResNet50 Model on ImageNet is ${acc}")
    assert(acc > 0.71)

  }
}
