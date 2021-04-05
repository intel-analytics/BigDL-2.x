package com.intel.analytics.zoo.serving.perf

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.pipeline.inference.InferenceModel
import com.intel.analytics.zoo.serving.utils.Supportive
import org.scalatest.{FlatSpec, Matchers}

class PerformanceSpec extends FlatSpec with Matchers with Supportive {
  val modelPath = "/home/litchy/models/"
  "TF resnet50 frozen 1_1" should "work" in {
    val model = new InferenceModel()
    model.doLoadTensorflow(s"${modelPath}tf_res50", "frozenModel", 1, 1, true)
    val data = Tensor[Float](1, 224, 224, 3).rand()
    for (a <- 0 to 5) {
      timing("batchsize 1 intra 1 predict"){
        model.doPredict(data)
      }
    }
  }
  "TF resnet50 frozen 1_2" should "work" in {
    val model = new InferenceModel()
    model.doLoadTensorflow(s"${modelPath}tf_res50", "frozenModel", 2, 1, true)
    val data = Tensor[Float](1, 224, 224, 3).rand()
    for (a <- 0 to 5) {
      timing("batchsize 1 intra 2 predict"){
        model.doPredict(data)
      }
    }
  }
  "TF resnet50 frozen 1_4" should "work" in {
    val model = new InferenceModel()
    model.doLoadTensorflow(s"${modelPath}tf_res50", "frozenModel", 4, 1, true)
    val data = Tensor[Float](1, 224, 224, 3).rand()
    for (a <- 0 to 5) {
      timing("batchsize 1 intra 4 predict"){
        model.doPredict(data)
      }
    }
  }
  "TF resnet50 frozen 4_1_1" should "work" in {
    val model = new InferenceModel()
    model.doLoadTensorflow(s"${modelPath}tf_res50", "frozenModel", 1, 1, true)
    val data = Tensor[Float](4, 224, 224, 3).rand()
    for (a <- 0 to 1000) {
      timing("batchsize 4 intra 1 predict"){
        model.doPredict(data)
      }
    }
  }
  "TF resnet50 frozen 4_2_1" should "work" in {
    val model = new InferenceModel()
    model.doLoadTensorflow(s"${modelPath}tf_res50", "frozenModel", 2, 1, true)
    val data = Tensor[Float](4, 224, 224, 3).rand()
    for (a <- 0 to 5) {
      timing("batchsize 4 intra 2 predict"){
        model.doPredict(data)
      }
    }
  }
  "TF resnet50 frozen 4_4_1" should "work" in {
    val model = new InferenceModel()
    model.doLoadTensorflow(s"${modelPath}tf_res50", "frozenModel", 4, 1, true)
    val data = Tensor[Float](4, 224, 224, 3).rand()
    for (a <- 0 to 1000) {
      timing("batchsize 4 intra 4 predict"){
        model.doPredict(data)
      }
    }
  }
  "TF resnet50 frozen 4_1_2" should "work" in {
    val model = new InferenceModel()
    model.doLoadTensorflow(s"${modelPath}tf_res50", "frozenModel", 1, 2, true)
    val data = Tensor[Float](4, 224, 224, 3).rand()
    for (a <- 0 to 100) {
      timing("batchsize 4 inter 2 predict"){
        model.doPredict(data)
      }
    }
  }
  "TF resnet50 frozen 4_1_4" should "work" in {
    val model = new InferenceModel()
    model.doLoadTensorflow(s"${modelPath}tf_res50", "frozenModel", 1, 4, true)
    val data = Tensor[Float](4, 224, 224, 3).rand()
    for (a <- 0 to 1000) {
      timing("batchsize 4 inter 4 predict"){
        model.doPredict(data)
      }
    }
  }
  "TF resnet50 savedModel 4_1" should "work" in {
    val model = new InferenceModel()
    model.doLoadTensorflow(s"${modelPath}tf_res50_sm", "savedModel", null, null, 1, 1)
    val data = Tensor[Float](4, 224, 224, 3).rand()
    for (a <- 0 to 5) {
      timing("batchsize 4 intra 1 predict"){
        model.doPredict(data)
      }
    }
  }
  "TF resnet50 savedModel 4_2" should "work" in {
    val model = new InferenceModel()
    model.doLoadTensorflow(s"${modelPath}tf_res50_sm", "savedModel", null, null, 2, 1)
    val data = Tensor[Float](4, 224, 224, 3).rand()
    for (a <- 0 to 5) {
      timing("batchsize 4 intra 2 predict"){
        model.doPredict(data)
      }
    }
  }
  "TF resnet50 savedModel 4_4" should "work" in {
    val model = new InferenceModel()
    model.doLoadTensorflow(s"${modelPath}tf_res50_sm", "savedModel", null, null, 4, 1)
    val data = Tensor[Float](4, 224, 224, 3).rand()
    for (a <- 0 to 5) {
      timing("batchsize 4 intra 4 predict"){
        model.doPredict(data)
      }
    }
  }
}
