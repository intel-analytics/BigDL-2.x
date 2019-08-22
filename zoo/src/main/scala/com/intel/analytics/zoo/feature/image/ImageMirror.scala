package com.intel.analytics.zoo.feature.image

import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat
import com.intel.analytics.bigdl.transform.vision.image.{FeatureTransformer, ImageFeature}
import org.opencv.core.Core

class Mirror() extends FeatureTransformer {
  override def transformMat(feature: ImageFeature): Unit = {
    Mirror.transform(feature.opencvMat(), feature.opencvMat())
  }
}

object Mirror {
  def apply(): Mirror = new Mirror()

  def transform(input: OpenCVMat, output: OpenCVMat): OpenCVMat ={
    Core.flip(input, output, -1)
    output
  }
}

class ImageMirror() extends ImageProcessing {
  private val internalCrop = new Mirror

  override def apply(prev: Iterator[ImageFeature]): Iterator[ImageFeature] = {
    internalCrop.apply(prev)
  }
}

object ImageMirror {
  def apply(): ImageMirror = new ImageMirror()
}
