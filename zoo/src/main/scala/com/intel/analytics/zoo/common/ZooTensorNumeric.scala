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

package com.intel.analytics.zoo.common

import com.intel.analytics.zoo.mkl.MKL


trait ZooTensorNumeric[@specialized(Float, Double) T] extends Serializable {
  def erf(n: Int, a: Array[T], aOffset: Int): Unit
}

object ZooTensorNumeric {
  implicit object ZooNumericFloat extends ZooTensorNumeric[Float]("Float") {
    override def erf(n: Int, a: Array[Float], aOffset: Int): Unit = {
      require(MKL.isMKLLoaded, "mkl isn't loaded")
      MKL.vsErf(n, a, aOffset)
    }
  }
}

