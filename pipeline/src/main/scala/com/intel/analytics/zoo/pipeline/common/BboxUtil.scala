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

package com.intel.analytics.zoo.pipeline.common

import com.intel.analytics.bigdl.tensor.Tensor
import org.apache.log4j.Logger

object BboxUtil {
  // inplace scale
  def scaleBBox(classBboxes: Tensor[Float], height: Float, width: Float): Unit = {
    if (classBboxes.nElement() == 0) return
    classBboxes.select(2, 1).apply1(_ * width)
    classBboxes.select(2, 2).apply1(_ * height)
    classBboxes.select(2, 3).apply1(_ * width)
    classBboxes.select(2, 4).apply1(_ * height)
  }


  val logger = Logger.getLogger(getClass)

  private def decodeSingleBbox(i: Int, priorBox: Tensor[Float], priorVariance: Tensor[Float],
    isClipBoxes: Boolean, bbox: Tensor[Float], varianceEncodedInTarget: Boolean,
    decodedBoxes: Tensor[Float]): Unit = {
    val x1 = priorBox.valueAt(i, 1)
    val y1 = priorBox.valueAt(i, 2)
    val x2 = priorBox.valueAt(i, 3)
    val y2 = priorBox.valueAt(i, 4)
    val priorWidth = x2 - x1
    require(priorWidth > 0)
    val priorHeight = y2 - y1
    require(priorHeight > 0)
    val pCenterX = (x1 + x2) / 2
    val pCenterY = (y1 + y2) / 2
    var decodeCenterX = 0f
    var decodeCenterY = 0f
    var decodeWidth = 0f
    var decodedHeight = 0f
    if (varianceEncodedInTarget) {
      // variance is encoded in target, we simply need to retore the offset
      // predictions.
      decodeCenterX = bbox.valueAt(i, 1) * priorWidth + pCenterX
      decodeCenterY = bbox.valueAt(i, 2) * priorHeight + pCenterY
      decodeWidth = Math.exp(bbox.valueAt(i, 3)).toFloat * priorWidth
      decodedHeight = Math.exp(bbox.valueAt(i, 4)).toFloat * priorHeight
    } else {
      // variance is encoded in bbox, we need to scale the offset accordingly.
      decodeCenterX = priorVariance.valueAt(i, 1) * bbox.valueAt(i, 1) * priorWidth + pCenterX
      decodeCenterY = priorVariance.valueAt(i, 2) * bbox.valueAt(i, 2) * priorHeight + pCenterY
      decodeWidth = Math.exp(priorVariance.valueAt(i, 3) * bbox.valueAt(i, 3)).toFloat * priorWidth
      decodedHeight = Math.exp(priorVariance.valueAt(i, 4) * bbox.valueAt(i, 4))
        .toFloat * priorHeight
    }
    decodedBoxes.setValue(i, 1, decodeCenterX - decodeWidth / 2)
    decodedBoxes.setValue(i, 2, decodeCenterY - decodedHeight / 2)
    decodedBoxes.setValue(i, 3, decodeCenterX + decodeWidth / 2)
    decodedBoxes.setValue(i, 4, decodeCenterY + decodedHeight / 2)
    if (isClipBoxes) {
      clipBoxes(decodedBoxes)
    }
  }

  def decodeBoxes(priorBoxes: Tensor[Float], priorVariances: Tensor[Float],
    isClipBoxes: Boolean, bboxes: Tensor[Float],
    varianceEncodedInTarget: Boolean, output: Tensor[Float] = null): Tensor[Float] = {
    require(priorBoxes.size(1) == priorVariances.size(1))
    require(priorBoxes.size(1) == bboxes.size(1))
    val numBboxes = priorBoxes.size(1)
    if (numBboxes > 0) {
      require(priorBoxes.size(2) == 4)
    }
    val decodedBboxes = if (output == null) Tensor[Float](numBboxes, 4)
    else output.resizeAs(priorBoxes)
    var i = 1
    while (i <= numBboxes) {
      decodeSingleBbox(i, priorBoxes,
        priorVariances, isClipBoxes, bboxes, varianceEncodedInTarget, decodedBboxes)
      i += 1
    }
    decodedBboxes
  }

  def clipBoxes(bboxes: Tensor[Float]): Tensor[Float] = {
    bboxes.cmax(0).apply1(x => Math.min(1, x))
  }

  def decodeBboxesAll(allLocPreds: Array[Array[Tensor[Float]]], priorBoxes: Tensor[Float],
    priorVariances: Tensor[Float], nClasses: Int, bgLabel: Int, clipBoxes: Boolean,
    varianceEncodedInTarget: Boolean, shareLocation: Boolean,
    output: Array[Array[Tensor[Float]]] = null)
  : Array[Array[Tensor[Float]]] = {
    val batch = allLocPreds.length
    val allDecodeBboxes = if (output == null) {
      val all = new Array[Array[Tensor[Float]]](batch)
      var i = 0
      while (i < batch) {
        all(i) = new Array[Tensor[Float]](nClasses)
        i += 1
      }
      all
    } else {
      require(output.length == batch)
      output
    }
    var i = 0
    while (i < batch) {
      val decodedBoxes = allDecodeBboxes(i)
      var c = 0
      while (c < nClasses) {
        // Ignore background class.
        if (shareLocation || c != bgLabel) {
          // Something bad happened if there are no predictions for current label.
          if (allLocPreds(i)(c).nElement() == 0) {
            logger.warn(s"Could not find location predictions for label $c")
          }
          val labelLocPreds = allLocPreds(i)(c)
          decodedBoxes(c) = decodeBoxes(priorBoxes, priorVariances, clipBoxes,
            labelLocPreds, varianceEncodedInTarget, labelLocPreds)
        }
        c += 1
      }
      allDecodeBboxes(i) = decodedBoxes
      i += 1
    }
    allDecodeBboxes
  }

  def getLocPredictions(loc: Tensor[Float], numPredsPerClass: Int, numClasses: Int,
    shareLocation: Boolean, locPredsBuf: Array[Array[Tensor[Float]]] = null)
  : Array[Array[Tensor[Float]]] = {
    // the outer array is the batch, each img contains an array of results, grouped by class
    val locPreds = if (locPredsBuf == null) {
      val out = new Array[Array[Tensor[Float]]](loc.size(1))
      var i = 0
      while (i < loc.size(1)) {
        out(i) = new Array[Tensor[Float]](numClasses)
        var c = 0
        while (c < numClasses) {
          out(i)(c) = Tensor[Float](numPredsPerClass, 4)
          c += 1
        }
        i += 1
      }
      out
    } else {
      locPredsBuf
    }
    var i = 0
    val locData = loc.storage().array()
    var locDataOffset = loc.storageOffset() - 1
    while (i < loc.size(1)) {
      val labelBbox = locPreds(i)
      var p = 0
      while (p < numPredsPerClass) {
        val startInd = p * numClasses * 4 + locDataOffset
        var c = 0
        while (c < numClasses) {
          val label = if (shareLocation) labelBbox.length - 1 else c
          val boxData = labelBbox(label).storage().array()
          val boxOffset = p * 4 + labelBbox(label).storageOffset() - 1
          val offset = startInd + c * 4
          boxData(boxOffset) = locData(offset)
          boxData(boxOffset + 1) = locData(offset + 1)
          boxData(boxOffset + 2) = locData(offset + 2)
          boxData(boxOffset + 3) = locData(offset + 3)
          c += 1
        }
        p += 1
      }
      locDataOffset += numPredsPerClass * numClasses * 4
      i += 1
    }
    locPreds
  }

  def getConfidenceScores(conf: Tensor[Float], numPredsPerClass: Int, numClasses: Int,
    confBuf: Array[Array[Tensor[Float]]] = null)
  : Array[Array[Tensor[Float]]] = {
    val confPreds = if (confBuf == null) {
      val out = new Array[Array[Tensor[Float]]](conf.size(1))
      var i = 0
      while (i < conf.size(1)) {
        out(i) = new Array[Tensor[Float]](numClasses)
        var c = 0
        while (c < numClasses) {
          out(i)(c) = Tensor[Float](numPredsPerClass)
          c += 1
        }
        i += 1
      }
      out
    }
    else confBuf
    val confData = conf.storage().array()
    var confDataOffset = conf.storageOffset() - 1
    var i = 0
    while (i < conf.size(1)) {
      val labelScores = confPreds(i)
      var p = 0
      while (p < numPredsPerClass) {
        val startInd = p * numClasses + confDataOffset
        var c = 0
        while (c < numClasses) {
          labelScores(c).setValue(p + 1, confData(startInd + c))
          c += 1
        }
        p += 1
      }
      confDataOffset += numPredsPerClass * numClasses
      i += 1
    }
    confPreds
  }

  def getPriorBboxes(prior: Tensor[Float], nPriors: Int): (Tensor[Float], Tensor[Float]) = {
    val array = prior.storage()
    val aOffset = prior.storageOffset()
    val priorBoxes = Tensor(array, aOffset, Array(nPriors, 4))
    val priorVariances = Tensor(array, aOffset + nPriors * 4, Array(nPriors, 4))
    (priorBoxes, priorVariances)
  }

}
