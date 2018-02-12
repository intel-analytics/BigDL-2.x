package com.intel.analytics.zoo.pipeline.utils

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.util.BoundingBox
import com.intel.analytics.bigdl.transform.vision.image.{FeatureTransformer, ImageFeature}

class ImageMeta(numClass: Int) extends FeatureTransformer {
  override def transformMat(feature: ImageFeature): Unit = {
    val meta = Tensor[Float](8 + numClass)
    meta.setValue(2, feature.getOriginalHeight)
    meta.setValue(3, feature.getOriginalWidth)
    meta.setValue(4, 3) // channel
    val windows = feature[BoundingBox](ImageFeature.boundingBox)
    meta.setValue(5, windows.x1)
    meta.setValue(6, windows.y1)
    meta.setValue(7, windows.x2)
    meta.setValue(8, windows.y2)
    feature(ImageMeta.imageMeta) = meta
  }
}

object ImageMeta {
  val imageMeta = "ImageMeta"
  def apply(numClass: Int): ImageMeta = new ImageMeta(numClass)
}