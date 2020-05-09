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
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.zoo.serving.DataType.DataTypeEnumVal

class PreProcessing(s: String) {
  val logger = Logger.getLogger(getClass)

  def bytesToTensor(dataType: DataTypeEnumVal, chwFlag: Boolean = true): Activity = {
    val b = java.util.Base64.getDecoder.decode(s)
    val result = dataType match {
      case DataType.IMAGE =>
        val mat = OpenCVMethod.fromImageBytes(b, Imgcodecs.CV_LOAD_IMAGE_UNCHANGED)
        val (height, width, channel) = (mat.height(), mat.width(), mat.channels())

        val data = new Array[Float](height * width * channel)
        OpenCVMat.toFloatPixels(mat, data)

        val imageTensor = Tensor[Float](data, Array(height, width, channel))

        if (chwFlag) {
          imageTensor.transpose(1, 3).transpose(2, 3).contiguous()
        } else {
          imageTensor
        }
      case DataType.TENSOR =>
        val alloc = new RootAllocator(Integer.MAX_VALUE)
        val MAX_ALLOC = 3 * 1024 * 1024 * 1024L
        val alloc4tensor = alloc.newChildAllocator("tensor", 0, MAX_ALLOC)
        val reader = new ArrowFileReader(new ByteArrayReadableSeekableByteChannel(b), alloc4tensor)

        val dataList = new ArrayBuffer[Tensor[Float]]

        try {
          while (reader.loadNextBatch) {
            val vsr = reader.getVectorSchemaRoot
            val arrowVector = vsr.getVector("0")
            val shapeLen = arrowVector.getObject(0).asInstanceOf[Float].toInt
            val dataLen = arrowVector.getObject(1).asInstanceOf[Float].toInt
            val shape = new Array[Int](shapeLen)
            val storage = new Array[Float](dataLen)

            for (i <- 0 until shapeLen) {
              shape(i) = arrowVector.getObject(i + 2).asInstanceOf[Float].toInt
            }

            for (i <- 0 until dataLen) {
              storage(i) = arrowVector.getObject(i + 2 + shapeLen).asInstanceOf[Float]
            }

            val dataTensor = Tensor[Float](storage, shape)
            dataList += dataTensor
          }
        } catch {
          case ex: IOException =>
            logger.warn(ex.getMessage)
        } finally {
          reader.close()
        }

        if (dataList.isEmpty) {
          Tensor[Float]()
        } else if (dataList.length == 1) {
          dataList(0)
        } else {
          T.array(dataList.toArray)
        }
    }

    result
  }
}
object PreProcessing {
  def apply(s: String, dataType: DataTypeEnumVal = DataType.IMAGE, chwFlag: Boolean = true,
            args: Array[Int] = Array()): Activity = {
    val cls = new PreProcessing(s)
    val t = cls.bytesToTensor(dataType, chwFlag)
    for (op <- args) {
      // new processing features add to here
    }
    t
  }
}
