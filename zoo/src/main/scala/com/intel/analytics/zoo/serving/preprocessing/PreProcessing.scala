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


import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat
import com.intel.analytics.zoo.feature.image.OpenCVMethod
import org.opencv.imgcodecs.Imgcodecs
import org.apache.log4j.Logger

import scala.collection.mutable.ArrayBuffer
import com.intel.analytics.bigdl.utils.{T, Table}
import com.intel.analytics.zoo.serving.http.Instances
import com.intel.analytics.zoo.serving.utils.Conventions
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import redis.clients.jedis.Jedis

class PreProcessing(chwFlag: Boolean = true,
                    redisHost: String = "localhost",
                    redisPort: Int = 6379,
                    jobName: String = Conventions.SERVING_STREAM_DEFAULT_NAME) {
  val logger = Logger.getLogger(getClass)

  var byteBuffer: Array[Byte] = null
  def decodeArrowBase64(key: String, s: String): Activity = {
    try {
      byteBuffer = java.util.Base64.getDecoder.decode(s)
      val instance = Instances.fromArrow(byteBuffer)

      val kvMap = instance.instances.flatMap(insMap => {
        val oneInsMap = insMap.map(kv =>
          if (kv._2.isInstanceOf[String]) {
            if (kv._2.asInstanceOf[String].contains("|")) {
              (kv._1, decodeString(kv._2.asInstanceOf[String]))
            }
            else {
              (kv._1, decodeImage(kv._2.asInstanceOf[String]))

            }
          }
          else {
            (kv._1, decodeTensor(kv._2.asInstanceOf[(
              ArrayBuffer[Int], ArrayBuffer[Float], ArrayBuffer[Int], ArrayBuffer[Int])]))
          }
        ).toList
        //      Seq(T(oneInsMap.head, oneInsMap.tail: _*))
        val arr = oneInsMap.map(x => x._2)
        Seq(T.array(arr.toArray))
      })
      kvMap.head
    } catch {
      case e: Exception =>
        logger.error(s"Preprocessing error, msg ${e.getMessage}")
        logger.error(s"Error stack trace ${e.getStackTrace.mkString("\n")}")
        null
    }
  }
  def decodeString(s: String): Tensor[String] = {
    val eleList = s.split("\\|")
    val tensor = Tensor[String](eleList.length)
    (1 to eleList.length).foreach(i => {
      tensor.setValue(i, eleList(i - 1))
    })
    tensor
  }

  def decodeImage(s: String, idx: Int = 0): Tensor[Float] = {
    byteBuffer = java.util.Base64.getDecoder.decode(s)
    val mat = OpenCVMethod.fromImageBytes(byteBuffer, Imgcodecs.CV_LOAD_IMAGE_UNCHANGED)
//    Imgproc.resize(mat, mat, new Size(224, 224))
    val (height, width, channel) = (mat.height(), mat.width(), mat.channels())

    val arrayBuffer = new Array[Float](height * width * channel)
    OpenCVMat.toFloatPixels(mat, arrayBuffer)

    val imageTensor = Tensor[Float](arrayBuffer, Array(height, width, channel))
    if (chwFlag) {
      imageTensor.transpose(1, 3)
        .transpose(2, 3).contiguous()
    } else {
      imageTensor
    }
  }
  def decodeTensor(info: (ArrayBuffer[Int], ArrayBuffer[Float],
    ArrayBuffer[Int], ArrayBuffer[Int])): Tensor[Float] = {
    val data = info._2.toArray
    val shape = info._1.toArray
    if (info._3.size == 0) {
      Tensor[Float](data, shape)
    } else {
      val indiceData = info._4.toArray
      val indiceShape = info._3.toArray
      var indice = new Array[Array[Int]](0)
      val colLength = indiceShape(1)
      var arr: Array[Int] = null
      (0 until indiceData.length).foreach(i => {
        if (i % colLength == 0) {
          arr = new Array[Int](colLength)
        }
        arr(i % colLength) = indiceData(i)
        if ((i + 1) % colLength == 0) {
          indice = indice :+ arr
        }
      })
      Tensor.sparse(indice, data, shape)
    }

  }

}
