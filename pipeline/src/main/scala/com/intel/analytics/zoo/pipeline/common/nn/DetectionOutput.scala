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

package com.intel.analytics.zoo.pipeline.common.nn

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Table
import com.intel.analytics.zoo.pipeline.common.BboxUtil
import com.intel.analytics.zoo.pipeline.common.nn.DetectionOutput._
import com.intel.analytics.zoo.pipeline.ssd.PostProcessParam
import org.apache.log4j.Logger

@SerialVersionUID(5253792953255433914L)
class DetectionOutput(param: PostProcessParam) extends AbstractModule[Table, Tensor[Float], Float] {
  @transient private var nms: Nms = _

  private def filterBboxes(decodedBboxes: Array[Tensor[Float]],
    confScores: Array[Tensor[Float]], indices: Array[Array[Int]],
    indicesNum: Array[Int]): Int = {
    var numDet = 0
    var c = 0
    while (c < param.nClasses) {
      if (c != param.bgLabel) {
        val scores = confScores(c)
        if (scores.nElement() == 0) {
          logger.warn(s"Could not find confidence predictions for label $c")
        }
        val label = if (param.shareLocation) decodedBboxes.length - 1 else c
        val bboxes = decodedBboxes(label)
        if (bboxes == null || bboxes.nElement() == 0) {
          logger.warn(s"Could not find locƒation predictions for label $label")
          return 0
        }
        indicesNum(c) = nms.nmsFast(scores, bboxes, param.nmsThresh,
          param.confThresh, indices(c), param.nmsTopk, normalized = true)

        numDet += indicesNum(c)
      }
      c += 1
    }
    if (param.keepTopK > -1 && numDet > param.keepTopK) {
      val scoreClassIndex = new Array[(Float, Int, Int)](numDet)
      var c = 0
      var count = 0
      while (c < indices.length) {
        var j = 0
        while (j < indicesNum(c)) {
          val idx = indices(c)(j)
          scoreClassIndex(count) = (confScores(c).valueAt(idx), c, idx)
          count += 1
          j += 1
        }
        indicesNum(c) = 0
        c += 1
      }
      // keep top k results per image
      val sortedPairs = scoreClassIndex.sortBy(x => -x._1)
      var i = 0
      while (i < param.keepTopK) {
        val label = sortedPairs(i)._2
        val idx = sortedPairs(i)._3
        indices(label)(indicesNum(label)) = idx
        indicesNum(label) += 1
        i += 1
      }
      param.keepTopK
    } else {
      numDet
    }
  }

  @transient var allLocPreds: Array[Array[Tensor[Float]]] = _
  @transient var allConfScores: Array[Array[Tensor[Float]]] = _
  @transient var allIndices: Array[Array[Array[Int]]] = _
  @transient var allIndicesNum: Array[Array[Int]] = _

  private def init(batch: Int, numLocClasses: Int, nPriors: Int): Unit = {
    var i = 0
    if (allLocPreds == null || allLocPreds.length < batch) {
      // the outer array is the batch, each img contains an array of results, grouped by class
      allLocPreds = new Array[Array[Tensor[Float]]](batch)
      allConfScores = new Array[Array[Tensor[Float]]](batch)
      allIndices = new Array[Array[Array[Int]]](batch)
      allIndicesNum = new Array[Array[Int]](batch)
      i = 0
      while (i < batch) {
        allLocPreds(i) = new Array[Tensor[Float]](numLocClasses)
        allConfScores(i) = new Array[Tensor[Float]](param.nClasses)
        allIndices(i) = new Array[Array[Int]](param.nClasses)
        allIndicesNum(i) = new Array[Int](param.nClasses)
        var c = 0
        while (c < numLocClasses) {
          allLocPreds(i)(c) = Tensor[Float](nPriors, 4)
          c += 1
        }
        c = 0
        while (c < param.nClasses) {
          allConfScores(i)(c) = Tensor[Float](nPriors)
          if (c != param.bgLabel) allIndices(i)(c) = new Array[Int](nPriors)
          c += 1
        }
        i += 1
      }

    } else {
      i = 0
      while (i < batch) {
        var c = 0
        while (c < numLocClasses) {
          allLocPreds(i)(c).resize(nPriors, 4)
          c += 1
        }
        c = 0
        while (c < param.nClasses) {
          allConfScores(i)(c).resize(nPriors)
          if (c != param.bgLabel && allIndices(i)(c).length < nPriors) {
            allIndices(i)(c) = new Array[Int](nPriors)
          }
          c += 1
        }
        i += 1
      }
    }
  }

  override def updateOutput(input: Table): Tensor[Float] = {
    if (nms == null) nms = new Nms()
    val loc = input[Tensor[Float]](1)
    val conf = input[Tensor[Float]](2)
    val prior = input[Tensor[Float]](3)
    val batch = loc.size(1)
    val numLocClasses = if (param.shareLocation) 1 else param.nClasses
    val nPriors = prior.size(3) / 4

    var i = 0

    init(batch, numLocClasses, nPriors)

    BboxUtil.getLocPredictions(loc, nPriors, numLocClasses, param.shareLocation,
      allLocPreds)

    BboxUtil.getConfidenceScores(conf, nPriors, param.nClasses, allConfScores)
    val (priorBoxes, priorVariances) = BboxUtil.getPriorBboxes(prior, nPriors)

    val allDecodedBboxes = BboxUtil.decodeBboxesAll(allLocPreds, priorBoxes, priorVariances,
      numLocClasses, param.bgLabel, false, param.varianceEncodedInTarget, param.shareLocation,
      allLocPreds)
    val numKepts = new Array[Int](batch)
    var maxDetection = 0

    i = 0
    while (i < batch) {
      val num = filterBboxes(allDecodedBboxes(i), allConfScores(i),
        allIndices(i), allIndicesNum(i))
      numKepts(i) = num
      maxDetection = Math.max(maxDetection, num)
      i += 1
    }
    // the first element is the number of detection numbers
    output = Tensor[Float](batch, 1 + maxDetection * 6)
    if (numKepts.sum > 0) {
      i = 0
      while (i < batch) {
        val outi = output(i + 1)
        var c = 0
        outi.setValue(1, numKepts(i))
        var offset = 2
        while (c < allIndices(i).length) {
          val indices = allIndices(i)(c)
          if (indices != null) {
            val indicesNum = allIndicesNum(i)(c)
            val locLabel = if (param.shareLocation) allDecodedBboxes(i).length - 1 else c
            val bboxes = allDecodedBboxes(i)(locLabel)
            var j = 0
            while (j < indicesNum) {
              val idx = indices(j)
              outi.setValue(offset, c)
              outi.setValue(offset + 1, allConfScores(i)(c).valueAt(idx))
              outi.setValue(offset + 2, bboxes.valueAt(idx, 1))
              outi.setValue(offset + 3, bboxes.valueAt(idx, 2))
              outi.setValue(offset + 4, bboxes.valueAt(idx, 3))
              outi.setValue(offset + 5, bboxes.valueAt(idx, 4))
              offset += 6
              j += 1
            }
          }
          c += 1
        }
        i += 1
      }
    }
    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[Float]): Table = {
    null
  }
}

object DetectionOutput {
  val logger = Logger.getLogger(getClass)

  def apply(param: PostProcessParam): DetectionOutput = new DetectionOutput(param)
}
