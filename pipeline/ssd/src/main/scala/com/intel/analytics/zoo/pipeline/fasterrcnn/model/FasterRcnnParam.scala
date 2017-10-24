/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.zoo.pipeline.fasterrcnn.model

import com.intel.analytics.bigdl.tensor.{Storage, Tensor}

abstract class FasterRcnnParam extends Serializable {
  val ratios: Array[Float]
  val scales: Array[Float]

  def anchorNum: Int = ratios.length * scales.length

  // Scales to use during training (can list multiple scales)
  // Each scale is the pixel size of an image"s shortest side
  var SCALES = Array(600)

  // Resize test images so that its width and height are multiples of ...
  val SCALE_MULTIPLE_OF = 1

  // Minibatch size (number of regions of interest [ROIs])
  val BATCH_SIZE = 3


  // Overlap threshold for a ROI to be considered background (class = 0 if
  // overlap in [LO, HI))
  val BG_THRESH_HI = 0.5
  val BG_THRESH_LO = 0.1

  // Overlap required between a ROI and ground-truth box in order for that ROI to
  // be used as a bounding-box regression training example
  val BBOX_THRESH = 0.5

  // Iterations between snapshots
  val SNAPSHOT_ITERS = 10000

  // Normalize the targets using "precomputed" (or made up) means and stdevs
  // (BBOX_NORMALIZE_TARGETS must also be true)
  var BBOX_NORMALIZE_TARGETS_PRECOMPUTED = true
  val BBOX_NORMALIZE_MEANS = Tensor(Storage(Array(0.0f, 0.0f, 0.0f, 0.0f)))
  val BBOX_NORMALIZE_STDS = Tensor(Storage(Array(0.1f, 0.1f, 0.2f, 0.2f)))

  // IOU >= thresh: positive example
  val RPN_POSITIVE_OVERLAP = 0.7
  // IOU < thresh: negative example
  val RPN_NEGATIVE_OVERLAP = 0.3
  // If an anchor statisfied by positive and negative conditions set to negative
  val RPN_CLOBBER_POSITIVES = false
  // Max number of foreground examples
  val RPN_FG_FRACTION = 0.5
  // Total number of examples
  val RPN_BATCHSIZE = 256
  // NMS threshold used on RPN proposals
  val RPN_NMS_THRESH = 0.7f
  // Number of top scoring boxes to keep before apply NMS to RPN proposals
  var RPN_PRE_NMS_TOP_N = 12000
  // Number of top scoring boxes to keep after applying NMS to RPN proposals
  var RPN_POST_NMS_TOP_N = 2000
  // Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
  val RPN_MIN_SIZE = 16
  // Deprecated (outside weights)
  val RPN_BBOX_INSIDE_WEIGHTS = Array(1.0f, 1.0f, 1.0f, 1.0f)
  // Give the positive RPN examples weight of p * 1 / {num positives}
  // and give negatives a weight of (1 - p)
  // Set to -1.0 to use uniform example weighting
  val RPN_POSITIVE_WEIGHT = -1.0f

  // Overlap threshold used for non-maximum suppression (suppress boxes with
  // IoU >= this threshold)
  val NMS = 0.3f

  // Apply bounding box voting
  val BBOX_VOTE = false

  val modelType: String
}




