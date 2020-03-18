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

import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.zoo.serving.utils.TensorUtils


/**
 * PostProssing
 * PostProcessing contains two steps
 * step 1 is filter, which is optional,
 * used to transform output tensor to type wanted
 * step 2 is to ndarray string, which is mandatory
 * to parse tensor into readable string
 * this string could be parsed by json in Python to a list
 * @param tensor
 */
class PostProcessing(tensor: Tensor[Float]) {
  var t: Tensor[Float] = tensor

  /**
   * Transform tensor into readable string,
   * could apply to any shape of tensor
   * @return
   */
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
  /**
   * TopN filter, take 1-D size (n) tensor as input
   * @param topN
   * @return 2-D size (topN, 2) tensor
   */
  def topN(topN: Int): String = {
    val list = TensorUtils.getTopN(topN, t)
    var res: String = ""
    res += "["
    (0 until list.size).foreach(i =>
      res += "[" + list(i)._1.toString + "," + list(i)._2.toString + "]"
    )
    res += "]"
    res
  }
}
object PostProcessing {
  def apply(t: Tensor[Float], filter: String = "None"): String = {
    val cls = new PostProcessing(t)
    if (filter != "None") {
      require(filter.last == ')',
        "please check your filter format, should be filter_name(filter_args)")
      require(filter.split("\\(").length == 2,
        "please check your filter format, should be filter_name(filter_args)")

      val filterType = filter.split("\\(").head
      val filterArgs = filter.split("\\(").last.dropRight(1).split(",")
      val res = filterType match {
        case "topN" =>
          require(filterArgs.length == 1, "topN filter only support 1 argument, please check.")
          cls.topN(filterArgs(0).toInt)
        case _ => ""
      }
      res
    }
    else {
      cls.tensorToNdArrayString()
    }
  }
}
