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

package com.intel.analytics.zoo.transform.vision.image3d.augmentation

import com.intel.analytics.zoo.transform.vision.image3d._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{DoubleType, FloatType, Tensor, Storage}
import scala.reflect.ClassTag

object Rotate {
  def apply(rotationAngles: Array[Double]): Rotate =
    new Rotate(rotationAngles)
}

class Rotate(rotationAngles: Array[Double])
  extends FeatureTransformer {
  private val List(yaw, pitch, roll) = rotationAngles.toList
  //  private val List(roll, pitch, yaw) = rotationAngles.toList
  private val rollDataArray = Array[Double](1, 0, 0,
    0, math.cos(roll), -math.sin(roll),
    0, math.sin(roll), math.cos(roll))

  private val pitchDataArray = Array[Double](math.cos(pitch), 0, math.sin(pitch),
    0, 1, 0,
    -math.sin(pitch), 0, math.cos(pitch))

  private val yawDataArray = Array[Double](math.cos(yaw), -math.sin(yaw), 0,
    math.sin(yaw), math.cos(yaw), 0,
    0, 0, 1)

  private val matSize = Array[Int](3, 3)

  private val rollDataTensor = Tensor[Double](rollDataArray, matSize)

  private val pitchDataTensor = Tensor[Double](pitchDataArray, matSize)

  private val yawDataTensor = Tensor[Double](yawDataArray, matSize)

  private val rotationTensor = yawDataTensor * pitchDataTensor * rollDataTensor

  override def transformTensor(tensor: Tensor[Float]): Tensor[Float] = {
    val depth = tensor.size(1)
    val height = tensor.size(2)
    val width = tensor.size(3)
    val dstData = Array.fill[Float](depth * height * width)(0f)
    val xc = (tensor.size(1) + 1) / 2.0
    val zc = (tensor.size(2) + 1) / 2.0
    val yc = (tensor.size(3) + 1) / 2.0
    var id, jd, kd: Double = 0
    var ii_0, ii_1, jj_0, jj_1, kk_0, kk_1: Int = 0

    val data = tensor.storage().array()
    for (i <- 1 to depth) {
      id = i
      for (k <- 1 to height) {
        kd = k
        for (j <- 1 to width) {
          var value = -1.0
          var ri, rj, rk, wi, wj, wk: Double = 0
          jd = j
          val coord = Tensor[Double](Array[Double](id - xc, jd - yc, kd - zc), Array[Int](3, 1))
          val rCoord = rotationTensor * coord
          val rData = rCoord.storage().array()
          ri = rData(0)
          rj = rData(1)
          rk = rData(2)

          ii_0 = math.floor(ri + xc).toInt
          jj_0 = math.floor(rj + yc).toInt
          kk_0 = math.floor(rk + zc).toInt

          ii_1 = ii_0 + 1
          jj_1 = jj_0 + 1
          kk_1 = kk_0 + 1

          wi = ri + xc - ii_0
          wj = rj + yc - jj_0
          wk = rk + zc - kk_0

          if (ii_1 == depth + 1 && wi < 0.5) ii_1 = ii_0
          else if (ii_1 >= depth + 1) value = 0.0
          if (jj_1 == width + 1 && wj < 0.5) jj_1 = jj_0
          else if (jj_1 >= width + 1) value = 0.0
          if (kk_1 == height + 1 && wk < 0.5) kk_1 = kk_0
          else if (kk_1 >= height + 1) value = 0.0

          if (ii_0 == 0 && wi > 0.5) ii_0 = ii_1
          else if (ii_0 < 1) value = 0.0
          if (jj_0 == 0 && wj > 0.5) jj_0 = jj_1
          else if (jj_0 < 1) value = 0.0
          if (kk_0 == 0 && wk > 0.5) kk_0 = kk_1
          else if (kk_0 < 1) value = 0.0

          if (value == -1.0) {
            value = (1 - wk) * (1 - wj) * (1 - wi) *
              (data((ii_0 - 1) * height * width + (kk_0 - 1) * width + jj_0 - 1).toDouble) +
              (1 - wk) * (1 - wj) * wi *
                (data((ii_1 - 1) * height * width + (kk_0 - 1) * width + jj_0 - 1).toDouble) +
              (1 - wk) * wj * (1 - wi) *
                (data((ii_0 - 1) * height * width + (kk_0 - 1) * width + jj_1 - 1).toDouble) +
              (1 - wk) * wj * wi *
                (data((ii_1 - 1) * height * width + (kk_0 - 1) * width + jj_1 -1).toDouble) +
              wk * (1 - wj) * (1 - wi) *
                (data((ii_0 - 1) * height * width + (kk_1 - 1) * width + jj_0 - 1).toDouble) +
              wk * (1 - wj) * wi *
                (data((ii_1 - 1) * height * width + (kk_1 - 1) * width + jj_0 - 1).toDouble) +
              wk * wj * (1 - wi) *
                (data((ii_0 - 1) * height * width + (kk_1 - 1) * width + jj_1 - 1).toDouble) +
              wk * wj * wi *
                (data((ii_1 - 1) * height * width + (kk_1 - 1) * width + jj_1 - 1).toDouble)
          }
          dstData((i - 1) * height * width + (k - 1) * width + j - 1) = value.toFloat
        }
      }
    }
    Tensor(storage = Storage[Float](dstData), storageOffset = 1, size = tensor.size())
  }
}
