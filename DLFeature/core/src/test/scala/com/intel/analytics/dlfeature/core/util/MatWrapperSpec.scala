/*
 * Copyright 2016 The BigDL Authors.
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

package com.intel.analytics.dlfeature.core.util

import com.intel.analytics.dlfeature.core.image.{Crop, Resize}
import org.scalatest.{FlatSpec, Matchers}

class MatWrapperSpec extends FlatSpec with Matchers {
  "toBytes" should "work properly" in {
    val resource = getClass().getClassLoader().getResource("image/000025.jpg")
    val img = MatWrapper.read(resource.getFile)
    Crop.transform(img, img, NormalizedBox(0, 0, 0.5f, 0.5f))
    Resize.transform(img, img, 300, 300)

    val img2 = MatWrapper.read(resource.getFile)
    Crop.transform(img2, img2, NormalizedBox(0, 0, 0.5f, 0.5f))
    val bytes = MatWrapper.toBytes(img2)
    val mat = MatWrapper.toMat(bytes)
    Resize.transform(mat, mat, 300, 300)

    val floats1 = new Array[Float](3 * 300 * 300)
    val floats2 = new Array[Float](3 * 300 * 300)
    val buf = new MatWrapper()
    MatWrapper.toFloatBuf(img, floats1, buf)
    MatWrapper.toFloatBuf(mat, floats2, buf)

    floats1.zip(floats2).foreach(x => {
      if (x._2 != x._1) {
        println(x._2 - x._1)
      }
    })
  }
}
