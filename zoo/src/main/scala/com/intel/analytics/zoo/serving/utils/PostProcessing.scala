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

package com.intel.analytics.zoo.serving.utils

import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import sun.util.resources.cldr.en.TimeZoneNames_en_SG

class PostProcessing(tensor: Tensor[Float]) {
  var t: Tensor[Float] = tensor
  def tensorToNdArrayString(): String = {
    val sizeArray = t.size()
    var strideArray = Array[Int]()
    val totalSize = {
      var res: Int = 1
      (0 until sizeArray.length).foreach(i => res *= sizeArray(i))
      res
    }


    (0 until sizeArray.length).foreach(i => {
      var res: Int = 1
      (0 to i).foreach(j => {
        res *= sizeArray(sizeArray.length - 1 - j)
      })
      strideArray = strideArray :+ res
    })
    val flatTensor = t.resize(totalSize).toArray()
    var str: String = ""
    (0 until flatTensor.length).foreach(i => {
      (0 until sizeArray.length).foreach(j => {
        if (i % strideArray(j) == 0) {
          str += "["
        }
      })
      str += flatTensor(i).toString
      (0 until sizeArray.length).foreach(j => {
        if ((i + 1) % strideArray(j) == 0) {
          str += "]"
        }
      })
      if (i != flatTensor.length - 1) {
        str += ","
      }
    })
    str
  }


  def getInfofromTensor(topN: Int, result: Tensor[Float]): String = {
    val outputSize = if (result.size(1) > topN) {
      topN
    } else {
      result.size(1)
    }

    val output = TensorUtils.getTopN(outputSize, result)
    var value: String = "{"
    (0 until outputSize - 1).foreach( j => {
      val tmpValue = "\"" + output(j)._1 + "\":\"" +
        output(j)._2.toString + "\","
      value += tmpValue
    })
    value += "\"" + output(outputSize - 1)._1 + "\":\"" +
      output(outputSize - 1)._2.toString
    value += "\"}"
    value
  }
  def topN(topN: Int): Tensor[Float] = {
    val list = TensorUtils.getTopN(topN, t)
    val res = Tensor[Float](list.size, 2)
    (0 until list.size).foreach(i => {
      res.setValue(i + 1, 1, list(i)._1)
      res.setValue(i + 1, 2, list(i)._2)
    })
    res
  }
}
object PostProcessing {
  def apply(t: Tensor[Float], filter: String = null): String = {
    val cls = new PostProcessing(t)
    require(filter.split(":").length == 2,
      "please check your filter format, should be filter_name:filter_args")
    val filterType = filter.split(":").head
    val fileterArgs = filter.split(":").last.split(",")
    cls.t = if (filterType == "topN") {
      require(fileterArgs.length == 1, "topN filter only support 1 argument, please check.")
      cls.topN(fileterArgs(0).toInt)
    } else {
      t
    }
    cls.tensorToNdArrayString()
  }
}
