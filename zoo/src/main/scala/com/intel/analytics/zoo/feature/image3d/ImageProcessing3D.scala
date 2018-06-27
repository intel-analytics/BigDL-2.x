package com.intel.analytics.zoo.feature.image3d

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.FeatureTransformer._
import com.intel.analytics.bigdl.transform.vision.image.ImageFeature
import com.intel.analytics.zoo.feature.image.{ImageProcessing, ImageSet}
import org.apache.log4j.Logger

/**
  * Created by jwang on 6/25/18.
  */
abstract class ImageProcessing3D extends ImageProcessing {

  /**
    * if true, catch the exception of the transformer to avoid crashing.
    * if false, interrupt the transformer when error happens
    */
  private var ignoreImageException: Boolean = false

  /**
    * catch the exception of the transformer to avoid crashing.
    */
  override def enableIgnoreException(): this.type = {
    ignoreImageException = true
    this
  }


  protected def transformTensor(tensor: Tensor[Float]): Tensor[Float] = {
    tensor
  }

  /**
    * transform feature
    *
    * @param feature ImageFeature3D
    * @return ImageFeature3D
    */
  override def transform(feature: ImageFeature): ImageFeature = {
    if (!feature.isInstanceOf[ImageFeature3D]) return feature
    else {
      transform(feature.asInstanceOf[ImageFeature3D])
    }
  }

  /**
    * transform feature
    *
    * @param feature ImageFeature3D
    * @return ImageFeature3D
    */
  def transform(feature: ImageFeature3D): ImageFeature3D = {
    try {
      if (!feature.isValid) return feature
      // change image to tensor
      val tensor = feature.asInstanceOf[ImageFeature3D][Tensor[Float]](ImageFeature.imageTensor)
      val out = transformTensor(tensor).clone()
      feature.update(ImageFeature.imageTensor, out)
      feature.update(ImageFeature.size, out.size())
    } catch {
      case e: Exception =>
        feature.isValid = false
        if (ignoreImageException) {
          val path = if (feature.contains(ImageFeature.uri)) feature(ImageFeature.uri) else ""
          logger.warn(s"failed ${path} in transformer ${getClass}")
          e.printStackTrace()
        } else {
          throw e
        }

    }
    feature
  }


  override def apply(imageSet: ImageSet): ImageSet = {
    imageSet.transform(this)
  }

}

object ImageProcessing3D {
  val logger = Logger.getLogger(getClass)
}
