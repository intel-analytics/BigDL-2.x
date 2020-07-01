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

import java.io.IOException

import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat
import com.intel.analytics.zoo.feature.image.OpenCVMethod
import org.opencv.imgcodecs.Imgcodecs
import org.apache.arrow.memory.RootAllocator
import org.apache.arrow.vector.ipc.ArrowFileReader
import org.apache.arrow.vector.util.ByteArrayReadableSeekableByteChannel
import org.apache.log4j.Logger

import scala.collection.mutable.ArrayBuffer
import com.intel.analytics.bigdl.utils.{T, Table}
import com.intel.analytics.zoo.serving.http.Instances
import com.intel.analytics.zoo.serving.preprocessing.DataType
import com.intel.analytics.zoo.serving.preprocessing.DataType.DataTypeEnumVal
import com.intel.analytics.zoo.serving.utils.SerParams
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

class PreProcessing(param: SerParams) {
  val logger = Logger.getLogger(getClass)

  var tensorBuffer: Array[Tensor[Float]] = null
  var arrayBuffer: Array[Array[Float]] = null

//  require(param.dataShape.length == param.dataType.length,
//    "Data shape length must be identical to data type length," +
//      "and match each other respectively")

  var byteBuffer: Array[Byte] = null

//  tensorBuffer = new Array[Tensor[Float]](param.dataShape.length)
//  createBuffer()

  def createBuffer(): Unit = {
    arrayBuffer = new Array[Array[Float]](param.dataShape.length)
    (0 until param.dataShape.length).foreach(idx => {
      val thisBuffer = new Array[Float](param.dataShape(idx).product)
      arrayBuffer(idx) = thisBuffer
    })
  }
  def decodeArrowBase64(s: String): Activity = {
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
        }).toList
//      Seq(T(oneInsMap.head, oneInsMap.tail: _*))
      val arr = oneInsMap.map(x => x._2)
      Seq(T.array(arr.toArray))
    })
    kvMap.head
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
    if (param.chwFlag) {
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


  def decodeBase64(s: String): Activity = {

    def decodeImage(idx: Int = 0): Tensor[Float] = {
      val mat = OpenCVMethod.fromImageBytes(byteBuffer, Imgcodecs.CV_LOAD_IMAGE_UNCHANGED)
      val (height, width, channel) = (mat.height(), mat.width(), mat.channels())

      OpenCVMat.toFloatPixels(mat, arrayBuffer(idx))

      val imageTensor = Tensor[Float](arrayBuffer(idx), Array(height, width, channel))
      if (param.chwFlag) {
        tensorBuffer(idx) = imageTensor.transpose(1, 3)
          .transpose(2, 3).contiguous()
      } else {
        tensorBuffer(idx) = imageTensor
      }
      imageTensor
    }
    def decodeTensor1(idx: Int, sparse: Boolean = false): Unit = {
      val alloc = new RootAllocator(Integer.MAX_VALUE)
      val MAX_ALLOC = 3 * 1024 * 1024 * 1024L
      val alloc4tensor = alloc.newChildAllocator("tensor", 0, MAX_ALLOC)
      val reader = new ArrowFileReader(
        new ByteArrayReadableSeekableByteChannel(byteBuffer), alloc4tensor)
      try {
        while (reader.loadNextBatch) {
          val vsr = reader.getVectorSchemaRoot
          val arrowVector = vsr.getVector("0")
          if (!sparse) {
            /**
             * decode dense Tensor
             */
            for (i <- 0 until arrowVector.getValueCount) {
              arrayBuffer(idx)(i) = arrowVector.getObject(i).asInstanceOf[Float]
            }
            tensorBuffer(idx) = Tensor[Float](arrayBuffer(idx), param.dataShape(idx))
          } else {
            /**
             * decode sparse Tensor
             * sparse Tensor has different schema
             * thus use another arrayBuffer
             */
            val elementNum = param.dataShape(idx)(0)
            val valueArray = new Array[Float](elementNum)
            val indicesBuffer = new Array[Int](arrowVector.getValueCount - elementNum)
            for (i <- 0 until elementNum) {
              valueArray(i) = arrowVector.getObject(i).asInstanceOf[Float].toInt
            }
            val indices = new Array[Array[Int]](param.dataShape.length)

            for (i <- elementNum until arrowVector.getValueCount) {
              indicesBuffer(i - elementNum) = arrowVector.getObject(i).asInstanceOf[Float].toInt
            }
            (0 until param.dataShape.length).foreach(dim => {
              val endIdx = (dim + 1) * indicesBuffer.length / param.dataShape.length
              indices(dim) = indicesBuffer.slice(dim, endIdx)
            })
            tensorBuffer(idx) = Tensor.sparse[Float](indices, valueArray, param.dataShape(idx))
          }


        }
      } catch {
        case ex: IOException =>
          logger.warn(ex.getMessage)
      } finally {
        reader.close()
      }
    }

    /**
     * Cluster Serving pre-defined protocol
     * "-" to split base64 encoded string
     */
    val encodedList = s.split("-")
    require(encodedList.length == tensorBuffer.length,
      "your input parameter length does not match your config.")
    (0 until encodedList.length).foreach(i => {
      byteBuffer = java.util.Base64.getDecoder.decode(encodedList(i))
//      param.dataType(i) match {
//        case DataType.IMAGE => decodeImage(i)
//        case DataType.TENSOR => decodeTensor1(i)
//        case DataType.SPARSETENSOR => decodeTensor1(i, true)
//      }
    })
    T.array(tensorBuffer)
  }
}
