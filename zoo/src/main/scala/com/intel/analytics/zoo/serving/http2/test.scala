package com.intel.analytics.zoo.serving.http2



import com.intel.analytics.zoo.pipeline.inference.InferenceModel
import com.intel.analytics.zoo.serving.http2.Models

object test{
  def main(args: Array[String]) {

    val instancesYaml = scala.io.Source.fromFile("/home/yansu/projects/test.yaml").mkString
    val modelInfoList = YamlUtil.fromYaml(classOf[Models], instancesYaml)
    modelInfoList.models.foreach(println)
    //println(YamlUtil.toYaml(YamlUtil.fromYaml(classOf[Models], instancesYaml)))
  }
}