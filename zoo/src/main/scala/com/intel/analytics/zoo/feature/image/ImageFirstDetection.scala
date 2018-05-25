package com.intel.analytics.zoo.feature.image

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.{FeatureTransformer, ImageFeature}
import com.intel.analytics.bigdl.utils.Table


class ImageFirstDetection extends FeatureTransformer {

  override protected def transformMat(feature: ImageFeature): Unit = {
    // only pick the first detection
    val output = feature.predict().asInstanceOf[Table]
    val numDetections = output[Tensor[Float]](1).valueAt(1).toInt
    val boxes = output[Tensor[Float]](2)
    val (ymin, xmin, ymax, xmax) =
      (boxes.valueAt(1, 1, 1), boxes.valueAt(1, 1, 2),
        boxes.valueAt(1, 1, 3), boxes.valueAt(1, 1, 4))
    val score = output[Tensor[Float]](3).valueAt(1, 1)
    val clas = output[Tensor[Float]](4).valueAt(1, 1)
    val pred = Tensor[Float](Array(clas, score, xmin, ymin, xmax, ymax), Array(1, 6))
    feature.update(ImageFeature.predict, pred)
  }

}

object ImageFirstDetection {
  def apply(): ImageFirstDetection = {
    new ImageFirstDetection()
  }
}
