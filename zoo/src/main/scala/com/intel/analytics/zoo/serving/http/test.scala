package com.intel.analytics.zoo.serving.http

object test{
  def main(args: Array[String]) {
//    val model = new InferenceModel(1)
//    model.doLoadOpenVINO(s"/home/yansu/projects/model/resnet_v1_50.xml",
//      s"/home/yansu/projects/model/resnet_v1_50.bin")
//    print(model.toString)
  val instancesJson =
    s"""{
       |"instances": [
       |   {
       |     "img": "yes"
       |   }
       |]
       |}
       |""".stripMargin

    printf(JsonUtil.toJson(JsonUtil.fromJson(classOf[Instances], instancesJson)))
  }
}