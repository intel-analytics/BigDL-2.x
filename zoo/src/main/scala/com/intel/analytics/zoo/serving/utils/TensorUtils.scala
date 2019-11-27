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

import com.intel.analytics.bigdl.tensor.Tensor

object TensorUtils {
  def getTopN(n: Int, t: Tensor[Float]): List[(Int, Float)] = {
    val arr = t.toArray().toList
    val idx = (0 until arr.size)
    val l = idx.zip(arr).toList

    def update(l: List[(Int, Float)], e: (Int, Float)): List[(Int, Float)] = {
      if (e._2 > l.head._2) (e :: l.tail).sortWith(_._2 < _._2) else l
    }

    l.drop(n).foldLeft(l.take(n).sortWith(_._2 < _._2))(update).
      sortWith(_._2 > _._2)
  }

  def main(args: Array[String]): Unit = {
    val list = List(3.0f, 2, 8, 2, 9, 1, 5, 5, 9, 1, 7, 3, 4)
    val t = Tensor[Float](4, 13)
    val b = t.select(1, 1)
    (0 until 13).map{ i =>
      t.setValue(i + 1, list(i))
    }
    val a = getTopN(5, t)
    a
  }

}
