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

package com.intel.analytics.zoo.pipeline.utils

import com.intel.analytics.bigdl.tensor.Tensor
import org.apache.log4j.Logger

object BboxUtil {
  val logger = Logger.getLogger(getClass)

  /**
   * Note that the output are stored in input deltas
   * @param boxes (N, 4)
   * @param deltas (N, 4a)
   * @return
   */
  def bboxTransformInv(boxes: Tensor[Float], deltas: Tensor[Float],
    normalized: Boolean = false): Tensor[Float] = {
    if (boxes.size(1) == 0) {
      return boxes
    }
    val output = Tensor[Float]().resizeAs(deltas).copy(deltas)
    require(boxes.size(2) == 4,
      s"boxes size ${boxes.size().mkString(",")} do not satisfy N*4 size")
    require(output.size(2) % 4 == 0,
      s"and deltas size ${output.size().mkString(",")} do not satisfy N*4a size")
    val boxesArr = boxes.storage().array()
    var offset = boxes.storageOffset() - 1
    val rowLength = boxes.stride(1)
    val deltasArr = output.storage().array()
    var i = 0
    val repeat = output.size(2) / boxes.size(2)
    var deltasoffset = output.storageOffset() - 1
    while (i < boxes.size(1)) {
      val x1 = boxesArr(offset)
      val y1 = boxesArr(offset + 1)
      val width = if (!normalized) boxesArr(offset + 2) - x1 + 1 else boxesArr(offset + 2) - x1
      val height = if (!normalized) boxesArr(offset + 3) - y1 + 1 else boxesArr(offset + 3) - y1
      var j = 0
      while (j < repeat) {
        j += 1
        // dx1*width + centerX
        val predCtrX = deltasArr(deltasoffset) * width + x1 + width / 2
        // dy1*height + centerY
        val predCtrY = deltasArr(deltasoffset + 1) * height + y1 + height / 2
        // exp(dx2)*width/2
        val predW = Math.exp(deltasArr(deltasoffset + 2)).toFloat * width / 2
        // exp(dy2)*height/2
        val predH = Math.exp(deltasArr(deltasoffset + 3)).toFloat * height / 2
        deltasArr(deltasoffset) = predCtrX - predW
        deltasArr(deltasoffset + 1) = predCtrY - predH
        deltasArr(deltasoffset + 2) = predCtrX + predW
        deltasArr(deltasoffset + 3) = predCtrY + predH
        deltasoffset += rowLength
      }
      offset += rowLength
      i += 1
    }
    output
  }

  def clipToWindows(windows: Tensor[Float], boxes: Tensor[Float]): Tensor[Float] = {
    val boxesArr = boxes.storage().array()
    var offset = boxes.storageOffset() - 1
    var i = 0
    val sw = windows.valueAt(1)
    val sh = windows.valueAt(2)
    val ew = windows.valueAt(3)
    val eh = windows.valueAt(4)
    while (i < boxes.size(1)) {
      boxesArr(offset) = Math.max(Math.min(boxesArr(offset), ew), sw)
      boxesArr(offset + 1) = Math.max(Math.min(boxesArr(offset + 1), eh), sh)
      boxesArr(offset + 2) = Math.max(Math.min(boxesArr(offset + 2), ew), sw)
      boxesArr(offset + 3) = Math.max(Math.min(boxesArr(offset + 3), eh), sh)
      offset += 4
      i += 1
    }
    boxes
  }


  def selectTensor(matrix: Tensor[Float], indices: Array[Int], dim: Int, indiceLen: Int = -1,
    out: Tensor[Float] = null): Tensor[Float] = {
    assert(dim == 1 || dim == 2)
    var i = 1
    val n = if (indiceLen == -1) indices.length else indiceLen
    if (matrix.nDimension() == 1) {
      val res = if (out == null) {
        Tensor[Float](n)
      } else {
        out.resize(n)
      }
      while (i <= n) {
        res.update(i, matrix.valueAt(indices(i - 1)))
        i += 1
      }
      return res
    }
    // select rows
    if (dim == 1) {
      val res = if (out == null) {
        Tensor[Float](n, matrix.size(2))
      } else {
        out.resize(n, matrix.size(2))
      }
      while (i <= n) {
        res.update(i, matrix(indices(i - 1)))
        i += 1
      }
      res
    } else {
      val res = if (out == null) {
        Tensor[Float](matrix.size(1), n)
      } else {
        out.resize(matrix.size(1), n)
      }
      while (i <= n) {
        var rid = 1
        val value = matrix.select(2, indices(i - 1))
        while (rid <= res.size(1)) {
          res.setValue(rid, i, value.valueAt(rid))
          rid += 1
        }
        i += 1
      }
      res
    }
  }

  def meshGrid(t1: Tensor[Float], t2: Tensor[Float]): (Tensor[Float], Tensor[Float]) = {
    val et1 = t1.reshape(Array(1, t1.nElement())).expand(Array(t2.nElement(), t1.nElement()))
    val et2 = t2.reshape(Array(t2.nElement(), 1)).expand(Array(t2.nElement(), t1.nElement()))
    (et1.contiguous(), et2.contiguous())
  }
}
