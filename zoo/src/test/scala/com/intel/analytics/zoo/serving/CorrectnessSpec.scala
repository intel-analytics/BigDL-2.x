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
import com.intel.analytics.zoo.serving.engine.InferenceSupportive
import com.intel.analytics.zoo.serving.http.{Instances, JsonUtil}
import com.intel.analytics.zoo.serving.postprocessing.PostProcessing
import com.intel.analytics.zoo.serving.utils.{ClusterServingHelper, SerParams}

import sys.env

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

    val b = FileUtils.readFileToByteArray(new File(path))
    val img = OpenCVMethod.fromImageBytes(b, Imgcodecs.CV_LOAD_IMAGE_COLOR)
    Imgproc.resize(img, img, new Size(224, 224))
    val matOfByte = new MatOfByte()
    Imgcodecs.imencode(".jpg", img, matOfByte)
    val dataStr = Base64.getEncoder.encodeToString(matOfByte.toArray)
    dataStr
  }


//  def runServingBg(): Future[Unit] = Future {
//    ClusterServing.run(configPath, redisHost, redisPort)
//  }
  "Cluster Serving result" should "be correct" in {

    ("wget -O /tmp/serving_val.tar http://10.239.45.10:8081" +
     "/repository/raw/analytics-zoo-data/imagenet_1k.tar").!
    "tar -xvf /tmp/serving_val.tar -C /tmp/".!
    val helper = new ClusterServingHelper(configPath)
    helper.initArgs()
    helper.dataShape = Array(Array(3, 224, 224))
    val param = new SerParams(helper)
    val model = helper.loadInferenceModel()
    val imagePath = "/tmp/imagenet_1k"
    val lsCmd = "ls " + imagePath

    val totalNum = (lsCmd #| "wc").!!.split(" +").filter(_ != "").head.toInt

    // enqueue image
    val f = new File(imagePath)
    val fileList = f.listFiles
    logger.info(s"${fileList.size} images about to enqueue...")

    val pre = new PreProcessing(param)
    pre.arrayBuffer = Array(new Array[Float](3 * 224 * 224))
    var predictMap = Map[String, String]()

    for (file <- fileList) {
      val dataStr = getBase64FromPath(file.getAbsolutePath)
      val instancesJson =
       s"""{
         |"instances": [
         |   {
         |     "img": "${dataStr}"
         |   }
         |]
         |}
         |""".stripMargin
      val instances = JsonUtil.fromJson(classOf[Instances], instancesJson)
      val inputBase64 = new String(java.util.Base64.getEncoder
       .encode(instances.toArrow()))
      val input = pre.decodeArrowBase64(inputBase64)
      val bInput = InferenceSupportive.batchInput(Seq(("", input)), param)
      val result = model.doPredict(bInput)
      val value = PostProcessing(result.toTensor[Float]
        .squeeze(1).select(1, 1), "topN(1)")
      val clz = value.split(",")(0).stripPrefix("[[")
      predictMap = predictMap + (file.getName -> clz)
    }
    ("rm -rf /tmp/" + imagePath + "*").!
    "rm -rf /tmp/serving_val_*".!
    "rm -rf /tmp/config.yaml".!

    // start check with txt file

    var cN = 0f
    var tN = 0f
    for (line <- Source.fromFile(imagePath + ".txt").getLines()) {
     val key = line.split(" ").head
     val cls = line.split(" ").tail(0)
     try {
       if (predictMap(key) == cls) {
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
