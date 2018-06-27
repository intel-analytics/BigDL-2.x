package com.intel.analytics.zoo.feature.image3d


import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.ImageFeature


/**
  * Created by jwang on 6/22/18.
  */
class ImageFeature3D extends ImageFeature {



  def this(tensor: Tensor[Float], label: Any, uri: String = null) {
    this
    update(ImageFeature.imageTensor, tensor)
    update(ImageFeature.size, tensor.size)
    if (null != uri) {
      update(ImageFeature.uri, uri)
    }
    if (null != label) {
      update(ImageFeature.label, label)
    }
  }

  /**
    * get current image size in [depth, height, width, channel]
    *
    * @return size array: [depth, height, width, channel]
    */
  def getImageSize(): Array[Int] = {
    apply[Array[Int]](ImageFeature.size)
  }

  /**
    * get current height
    */
  def getDepth(): Int = getImageSize()(0)

  /**
    * get current height
    */
  override def getHeight(): Int = getImageSize()(1)

  /**
    * get current width
    */
  override def getWidth(): Int = getImageSize()(2)

  /**
    * get current channel
    */
  override def getChannel(): Int = getImageSize()(3)


  override def clone(): ImageFeature3D = {
    val imageFeature = new ImageFeature3D()
    keys().foreach(key => {
      imageFeature.update(key,this.apply(key))
    })
    imageFeature.isValid = isValid
    imageFeature
  }
}


object ImageFeature3D {

  def apply(tensor: Tensor[Float], uri: String = null, label: Any = null)
  : ImageFeature3D = new ImageFeature3D(tensor, label, uri)

  def apply(): ImageFeature3D = new ImageFeature3D()
}

