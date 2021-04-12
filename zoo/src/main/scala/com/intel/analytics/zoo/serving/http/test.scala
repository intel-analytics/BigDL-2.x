package com.intel.analytics.zoo.serving.http

import java.util

import com.intel.analytics.zoo.pipeline.inference.{InferenceModel, JTensor}
import java.util.{List => JList}
object test{
  def main(args: Array[String]) {
    val model = new InferenceModel(1)
    model.doLoadOpenVINO(s"/home/yansu/projects/model/resnet_v1_50.xml",
      s"/home/yansu/projects/model/resnet_v1_50.bin")
    print(model.toString)
    val str : List[Float] = List[Float](1.1f, 2)


    val data = new util.ArrayList[JList[JTensor]]()
    val dataj = new Array[Float] (1)
    dataj.update(0,10)
    val shapej= new Array[Int] (1)
    shapej.update(0,10)
    val testTensor = new JTensor(str.toArray, shapej, false)
    data.add(util.Arrays.asList({testTensor}))
    print(model.doPredict(inputs = data))
  }
}