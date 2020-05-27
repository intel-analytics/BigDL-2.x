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

import com.intel.analytics.zoo.serving.preprocessing.DataType

object ConfigUtils {
  def parseShape(shape: String): Array[Array[Int]] = {
    val shapeListStr = shape.
      split("""\[\[|\]\]|\],\s*\[""").filter(x => x != "")
    var shapeList = new Array[Array[Int]](shapeListStr.length)
    (0 until shapeListStr.length).foreach(idx => {
      val arr = shapeListStr(idx).stripPrefix("[").stripSuffix("]").split(",")
      val thisShape = new Array[Int](arr.length)
      (0 until arr.length).foreach(i => {
        thisShape(i) = arr(i).trim.toInt
      })
      shapeList(idx) = thisShape
    })
    shapeList
  }
  def parseType(tp: String): Array[DataType.DataTypeEnumVal] = {
    val arr = tp.stripPrefix("[").stripSuffix("]").split(",")
    val thisType = new Array[DataType.DataTypeEnumVal](arr.length)
    (0 until arr.length).foreach(idx => {
      thisType(idx) = arr(idx).trim match {
        case "image" => DataType.IMAGE
        case "tensor" => DataType.TENSOR
        case "sparse_tensor" => DataType.SPARSETENSOR
      }

    })
    thisType
  }
}
