package com.intel.analytics.zoo.pipeline.utils

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.augmentation.Resize
import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat
import com.intel.analytics.bigdl.transform.vision.image.util.BoundingBox
import com.intel.analytics.bigdl.transform.vision.image.{FeatureTransformer, ImageFeature}
import com.intel.analytics.bigdl.utils.Table

class UnmodeDetection() extends FeatureTransformer {
  override def transformMat(feature: ImageFeature): Unit = {
    val output = feature[Table](ImageFeature.predict)
    val detections = output[Tensor[Float]](1)
    val mrcnnMask = output[Tensor[Float]](4)
    val windows = feature[BoundingBox](ImageFeature.boundingBox)
    // Extract boxes, class_ids, scores, and class-specific masks
    val boxes = detections.narrow(2, 1, 4)
    val classIds = detections.select(2, 5)
    val scores = detections.select(2, 6)
    val masks = (1 to boxes.size(1)).map(i => {
      mrcnnMask(i)(classIds.valueAt(i).toInt)
    }).toArray
    val hScale = feature.getOriginalHeight / (windows.y2 - windows.y1)
    val wScale = feature.getOriginalWidth / (windows.x2 - windows.x1)
    val scale = Math.min(hScale, wScale)
    // Translate bounding boxes to image domain
    boxes.narrow(2, 1, 1).add(-windows.x1)
    boxes.narrow(2, 2, 1).add(-windows.y1)
    boxes.narrow(2, 3, 1).add(-windows.x1)
    boxes.narrow(2, 4, 1).add(-windows.y1)
    boxes.mul(scale)

    val fullMasks = (1 to boxes.size(1)).map(i => {
      unmodeMask(masks(i - 1), boxes(i), feature.getOriginalHeight, feature.getOriginalWidth)
    }).toArray
    classIds.apply1(_ - 1)
    feature("unmode") = (boxes, classIds, scores, fullMasks)
  }

  private def unmodeMask(mask: Tensor[Float], bbox: Tensor[Float], height: Int, width: Int)
  : Tensor[Float] = {
    val x1 = bbox.valueAt(1).toInt
    val y1 = bbox.valueAt(2).toInt
    val x2 = bbox.valueAt(3).toInt
    val y2 = bbox.valueAt(4).toInt
    if (x2 <= x1 || y2 <= y1) {
      return null
    }
    val mat = OpenCVMat.fromTensor(mask.reshape(Array(mask.size(1), mask.size(2), 1)))
    Resize.transform(mat, mat, x2 - x1, y2 - y1)
    val out = OpenCVUtil.toTensor(mat)
    out.apply1(x => {
      if (x >= 0.5) 1 else 0
    })
    // Put the mask in the right location.
    val full = Tensor[Float](height, width)
    full.narrow(1, y1 + 1, y2 - y1).narrow(2, x1 + 1, x2 - x1).copy(out)
    full
  }
}

object UnmodeDetection {
  def apply(): UnmodeDetection = new UnmodeDetection()
}
