package com.intel.analytics.zoo.serving.utils

import com.intel.analytics.bigdl.dataset.{MiniBatch, SampleToMiniBatch}
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image.ImageFeature
import com.intel.analytics.zoo.feature.image.{DistributedImageSet, ImageSet}
import com.intel.analytics.zoo.models.image.imageclassification.{LabelOutput, LabelReader}
import com.intel.analytics.zoo.pipeline.inference.InferenceModel
import com.intel.analytics.zoo.serving.utils.{ClusterServingHelper, Result}


object ImageClassification {
  def getResult(img: ImageSet, model: InferenceModel, loader: ClusterServingHelper) = {
    implicit val ev = TensorNumeric.NumericFloat

    val batch = img.toDataSet() -> SampleToMiniBatch(loader.batchSize)

    val predicts = batch.toDistributed().data(false).flatMap { miniBatch =>
      val predict = model.doPredict(miniBatch
        .getInput.toTensor)

      predict.toTensor.squeeze.split(1).asInstanceOf[Array[Activity]]
    }

    if (img.isDistributed()) {

      val zipped = img.toDistributed().rdd.zip(predicts)
      val zk = zipped.collect()
      zipped.map(tuple => {
        tuple._1(ImageFeature.predict) = tuple._2
      }).collect()
    }


    // Transform prediction into Labels and probs
    val labelOutput = LabelOutput(LabelReader.apply("IMAGENET"))
    //        print(labelOutput(imageSet).isInstanceOf[DistributedImageSet])
    //        print(labelOutput(imageSet).isInstanceOf[LocalImageSet])

    val topN = loader.topN

    val result = labelOutput(img).toDistributed().rdd.map { f =>
      val probs = f("probs").asInstanceOf[Array[Float]]
      var value: String = "top-n_class_probability: {"
      for (i <- 0 until topN - 1) {
        value = value + probs(i).toString + ","
      }
      value = value + probs(topN - 1) + "}"
      // remember use case class here
      // this is the only key-value pair support
      // if you use tuple, you will get key of null
      Result(f("uri"), value)

    }
    result
  }
}
