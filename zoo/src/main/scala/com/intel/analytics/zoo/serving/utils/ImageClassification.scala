package com.intel.analytics.zoo.serving.utils

import com.intel.analytics.bigdl.dataset.{MiniBatch, SampleToMiniBatch}
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image.ImageFeature
import com.intel.analytics.zoo.feature.image.{DistributedImageSet, ImageSet}
import com.intel.analytics.zoo.models.image.imageclassification.{LabelOutput, LabelReader}
import com.intel.analytics.zoo.pipeline.inference.{FloatModel, InferenceModel}
import com.intel.analytics.zoo.serving.utils.{ClusterServingHelper, Result}


object ImageClassification {
  def getResult(img: ImageSet, model: InferenceModel, loader: ClusterServingHelper) = {
    implicit val ev = TensorNumeric.NumericFloat

    val batch = img.toDataSet() -> SampleToMiniBatch(loader.batchSize)

    val modelType = loader.modelType

    val t = batch.toDistributed().data(false).collect()
    val c = batch.toDistributed().data(false).count()
    val predicts = batch.toDistributed().data(false).flatMap { miniBatch =>
      val batchTensor = if (modelType == "openvino") {
        miniBatch.getInput.toTensor.addSingletonDimension()
      }
      else if (modelType == "tensorflow") {
        miniBatch.getInput.toTensor.transpose(2,3)
            .transpose(3,4).contiguous()
      }
      else {
        miniBatch.getInput.toTensor
      }
      // this is hard code to test TensorFLow, NHWC

//      batchTensor.resize(4, 224,224,3)

//      val predict = model.getOriginalModel.asInstanceOf[FloatModel].model.forward(batchTensor)
      val predict = model.doPredict(batchTensor)

      val cc = predict.toTensor
//      predict
//      val aa = predict.toTensor.squeeze
//      val a = predict.toTensor.squeeze.split(1).asInstanceOf[Array[Activity]]
      if (predict.toTensor.dim == 1) {
        Array(predict.toTensor.asInstanceOf[Activity])
      }
      else {
        predict.toTensor.squeeze.split(1).asInstanceOf[Array[Activity]]
      }
    }
//    predicts.count()
    if (img.isDistributed()) {
//      println("img rdd is " + img.toDistributed().rdd.count())
//      println("predict rdd is " + predicts.count())
//      val a = img.toDistributed().rdd.collect()
//      val b = predicts.collect()
      val zipped = img.toDistributed().rdd.zip(predicts)
//      val zk = zipped.collect()
      zipped.map(tuple => {
        tuple._1(ImageFeature.predict) = tuple._2
      }).collect()
    }

//
    // Transform prediction into Labels and probs
    var dummyMap: Map[Int, String] = Map()
    for (i <- 0 to 5000) {
      dummyMap += (i -> ("Class No." + i.toString))
    }


//    val labelOutput = LabelOutput(LabelReader.apply("IMAGENET"))
    val labelOutput = LabelOutput(dummyMap)

    //        print(labelOutput(imageSet).isInstanceOf[DistributedImageSet])
    //        print(labelOutput(imageSet).isInstanceOf[LocalImageSet])

    val topN = loader.topN

    val result = labelOutput(img).toDistributed().rdd.map { f =>
      val probs = f("probs").asInstanceOf[Array[Float]]
      val cls = f("classes").asInstanceOf[Array[String]]
      var value: String = "{top-n_class: {"
      for (i <- 0 until topN - 1) {
        value = cls(i) + ": "
        value = value + probs(i).toString + ","
      }
      value = value + cls(topN - 1) + ": " + probs(topN - 1) + "}}"
      println(value)
      // remember use case class here
      // this is the only key-value pair support
      // if you use tuple, you will get key of null
      Result(f("uri"), value)

    }
    result
  }
}
