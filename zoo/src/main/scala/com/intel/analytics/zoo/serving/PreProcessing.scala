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

import scala.collection.mutable.ArrayBuffer
import scala.util.control.Breaks._
import com.intel.analytics.bigdl.utils.T

class PreProcessing(s: String) {
  def bytesToTensor(dataType: String, chwFlag: Boolean = true): Activity = {
    val b = java.util.Base64.getDecoder.decode(s)
    val result = dataType match {
      case "image" =>
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
      case "tensor" =>
        val alloc = new RootAllocator(Integer.MAX_VALUE)
        val MAX_ALLOC = 3 * 1024 * 1024 * 1024L
        val alloc4tensor = alloc.newChildAllocator("tensor", 0, MAX_ALLOC)
        val reader = new ArrowFileReader(new ByteArrayReadableSeekableByteChannel(b), alloc4tensor)

        val dataList = new ArrayBuffer[Tensor[Float]]

        try {
          while (reader.loadNextBatch) {
            var shape = new ArrayBuffer[Int]()
            val vsr = reader.getVectorSchemaRoot
            val accessor = vsr.getVector("0")
            var idx = 0
            val valueCount = accessor.getValueCount
            breakable {
              while (idx < valueCount) {
                val data = accessor.getObject(idx).asInstanceOf[Float].toInt
                idx += 1
                if (data == -1) {
                  break
                }
                shape += data
              }
            }
            val storage = new Array[Float](valueCount - idx)

            for (i <- idx until valueCount) {
              storage(i-idx) = accessor.getObject(i).asInstanceOf[Float]
            }

            val dataTensor = Tensor[Float](storage, shape.toArray)
            dataList += dataTensor
          }
        } catch {
          case ex: IOException =>
            // TODO
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
  def apply(s: String, dataType: String = "image", chwFlag: Boolean = true,
            args: Array[Int] = Array()): Activity = {
    val cls = new PreProcessing(s)
//    val startTime: Long = System.currentTimeMillis
    val t = cls.bytesToTensor(dataType, chwFlag)
//    val endTime: Long = System.currentTimeMillis
//    val diff = endTime - startTime
//    System.out.println("time ：" + diff + "ms")
    for (op <- args) {
      // new processing features add to here
    }
    t
  }
}
