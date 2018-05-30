package com.intel.analytics.zoo.pipeline.inference

import com.intel.analytics.bigdl.optim.LocalPredictor
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.tensor.Tensor
import scala.collection.JavaConverters._

case class FloatInferenceModel(
  model:     AbstractModule[Activity, Activity, Float],
  predictor: LocalPredictor[Float]) extends InferenceSupportive {

  def forward(data: Tensor[Float])(implicit om: Manifest[Float]): Array[Float] = {
    model.forward(data).asInstanceOf[Tensor[Float]].storage().array()
  }

  def predict(samples: Array[Sample[Float]]): Array[Int] = {
    predictor.predictClass(samples)
  }

  def predict(input: java.util.List[java.util.List[java.lang.Float]]): java.util.List[java.lang.Float] = {
    timing(s"predict for $input") {
      val tensor = transferInferenceInputToTensor(input)
      val result = model.forward(tensor).asInstanceOf[Tensor[Float]].storage.array.toList
        .asJava.asInstanceOf[java.util.List[java.lang.Float]]
      result
    }
  }
}