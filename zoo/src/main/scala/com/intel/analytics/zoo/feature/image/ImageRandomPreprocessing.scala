package com.intel.analytics.zoo.feature.image

import com.intel.analytics.bigdl.transform.vision.image.{FeatureTransformer, ImageFeature, augmentation}

/**
 * Randomly apply the preprocessing to some of the input ImageFeatures, with probability specified.
 *
 * @param preprocessing preprocessing to apply randomness
 * @param prob probability to apply the transform action
 */
class ImageRandomPreprocessing(
    preprocessing: ImageProcessing,
    prob: Double
  ) extends ImageProcessing  {

  private val internalRandomTransformer = new augmentation.RandomTransformer(preprocessing, prob)
  override def apply(prev: Iterator[ImageFeature]): Iterator[ImageFeature] = {
    internalRandomTransformer.apply(prev)
  }

  override def transformMat(feature: ImageFeature): Unit = {
    internalRandomTransformer.transformMat(feature)
  }

}

object ImageRandomPreprocessing {
  def apply(preprocessing: ImageProcessing, prob: Double): ImageRandomPreprocessing =
    new ImageRandomPreprocessing(preprocessing, prob)
}