package com.intel.analytics.zoo.serving.utils

import com.intel.analytics.bigdl.dataset.MiniBatch
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image.ImageFeature
import com.intel.analytics.zoo.feature.image.ImageSet
import com.intel.analytics.zoo.models.image.objectdetection.{LabelReader, ScaleDetection, Visualizer}
import com.intel.analytics.zoo.pipeline.inference.InferenceModel
import com.intel.analytics.zoo.serving.utils.{ClusterServingHelper, Result}


object ObjectDetection {
  def getResult(img: ImageSet, model: InferenceModel, loader: ClusterServingHelper) = {
    implicit val ev = TensorNumeric.NumericFloat

    val predicts = img.toDistributed().rdd.map { img =>
      val predict = model
        .doPredict(img.apply[Tensor[Float]](ImageFeature.imageTensor))

      img(ImageFeature.predict) = predict.toTensor[Float].apply(1)
      img

    }
    val labelMap = LabelReader.apply("COCO")
    val boxedPredict = ImageSet.rdd(predicts) -> ScaleDetection()

    val visualizer = Visualizer(labelMap, encoding = "jpg")
    val visualized = visualizer(boxedPredict).toDistributed()
    val result = visualized.rdd.map { f =>

      Result(f("uri"), f[Array[Byte]](Visualizer.visualized).toString)
    }
    result
  }
}
