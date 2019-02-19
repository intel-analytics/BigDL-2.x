package com.intel.analytics.zoo.feature.image

import com.intel.analytics.bigdl.transform.vision.image.{FeatureTransformer, ImageFeature, augmentation}
import com.intel.analytics.bigdl.transform.vision.image.augmentation.Crop
import com.intel.analytics.bigdl.transform.vision.image.label.roi.{BatchSampler, RandomSampler, RoiLabel, RoiProject}
import com.intel.analytics.bigdl.transform.vision.image.util.BoundingBox
import com.intel.analytics.bigdl.utils.RandomGenerator._
import org.opencv.core.Mat

import scala.collection.mutable.ArrayBuffer

/**
  * Random sample a bounding box given some constraints and crop the image
  * This is used in SSD training augmentation
  */
class ImageRandomSampler extends ImageProcessing {
  private val internalSampler = RandomSampler()
  override def apply(prev: Iterator[ImageFeature]): Iterator[ImageFeature] = {
    internalSampler.apply(prev)
  }

//  override def transformMat(feature: ImageFeature): Unit = {
//    internalSampler.transformMat(feature)
//  }
}

object ImageRandomSampler {
  def apply(): ImageRandomSampler = {
    new ImageRandomSampler()
  }
}
