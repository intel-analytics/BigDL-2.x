//package com.intel.analytics.zoo.serving.http
//
//import com.codahale.metrics.Timer
//import com.intel.analytics.bigdl.tensor.Tensor
//import com.intel.analytics.zoo.pipeline.inference.InferenceModel
//import com.intel.analytics.zoo.serving.http.test.model
//import com.intel.analytics.zoo.serving.serialization.JsonInputDeserializer
//
//object test extends App{
//  def timing[T](name: String)(timers: Timer*)(f: => T): T = {
//    val begin = System.nanoTime()
//    val contexts = timers.map(_.time())
//    val result = f
//    contexts.map(_.stop())
//    val end = System.nanoTime()
//    val cost = (end - begin)
//    println(s"$name time elapsed [${cost / 1e6} ms]")
//    result
//  }
//
//
//  val model = new InferenceModel(1)
//  model.doLoadTensorflow("/root/yansu/models/freezed", "frozenModel")
//  val input = scala.io.Source.fromFile("/root/yansu/inference_model_input/dien.txt").mkString
//  val activity = JsonInputDeserializer.deserialize(input)
//  // Share Tensor Storage
//  0.to(1000).foreach(_ => model.doPredict(activity))
//
//  timing("predict activity time avg")(){
//    0.to(1000).foreach(_ => model.doPredict(activity))
//  }
//
//}
//
//object test2 extends App{
//  def timing[T](name: String)(timers: Timer*)(f: => T): T = {
//    val begin = System.nanoTime()
//    val contexts = timers.map(_.time())
//    val result = f
//    contexts.map(_.stop())
//    val end = System.nanoTime()
//    val cost = (end - begin)
//    println(s"$name time elapsed [${cost / 1e6} ms]")
//    result
//  }
//
//
//  val model = new InferenceModel(1)
//  model.doLoadTensorflow("/root/yansu/models/freezed", "frozenModel")
//  val content = scala.io.Source.fromFile("/root/yansu/inference_model_input/dien.txt").mkString
//  val instances = JsonUtil.fromJson(classOf[Instances], content)
//  val features = List[String] ("a", "b", "c", "d", "e", "f", "g")
//  val activities = instances.makeActivities(features.toArray)
//  0.to(1000).foreach(_ => model.doPredict(activities.head))
//  timing("predict activity time avg new")() {
//    0.to(1000).foreach(_ => model.doPredict(activities.head))
//  }
//
//}
