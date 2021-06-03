package com.intel.analytics.zoo.serving.http

import com.codahale.metrics.Timer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.pipeline.inference.InferenceModel
import com.intel.analytics.zoo.serving.serialization.JsonInputDeserializer

object test extends App{
  def timing[T](name: String)(timers: Timer*)(f: => T): T = {
    val begin = System.nanoTime()
    val contexts = timers.map(_.time())
    val result = f
    contexts.map(_.stop())
    val end = System.nanoTime()
    val cost = (end - begin)
    println(s"$name time elapsed [${cost / 1e6} ms]")
    result
  }


  val model = new InferenceModel()
  model.doLoadTensorflow("/home/yansu/projects/models/freezed", "frozenModel")
  val input = scala.io.Source.fromFile("/home/yansu/projects/inference_model_input/dien.txt").mkString
  val activity = JsonInputDeserializer.deserialize(input)
  // Share Tensor Storage


}
