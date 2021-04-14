package com.intel.analytics.zoo.serving.http2

import com.intel.analytics.zoo.serving.http.FrontEndApp.timing
import com.intel.analytics.zoo.serving.http._
import akka.actor.ActorRef

object test {
  def main(args: Array[String]) {
    val content =
      """{
  "instances" : [ {
    "intScalar" : 12345,
    "floatScalar" : 3.14159,
    "stringScalar" : "hello, world. hello, zoo.",
    "intTensor" : [ 7756, 9549, 1094, 9808, 4959, 3831, 3926, 6578, 1870, 1741 ],
    "floatTensor" : [ 0.6804766, 0.30136853, 0.17394465, 0.44770062, 0.20275897 ],
    "stringTensor" : [ "come", "on", "united" ],
    "intTensor2" : [ [ 1, 2 ], [ 3, 4 ], [ 5, 6 ] ],
    "floatTensor2" : [ [ [ 0.2, 0.3 ], [ 0.5, 0.6 ] ], [ [ 0.2, 0.3 ], [ 0.5, 0.6 ] ] ],
    "stringTensor2" : [ [ [ [ "come", "on", "united" ], [ "come", "on", "united" ] ] ] ],
    "sparseTensor" : {
      "shape" : [ 100, 10000, 10 ],
      "data" : [ 0.2, 0.5, 3.45, 6.78 ],
      "indices" : [ [ 1, 1, 1 ], [ 2, 2, 2 ], [ 3, 3, 3 ], [ 4, 4, 4 ] ]
    },
    "image": "/"
  } ]
}"""

    try {
      val servableManager = new ServableManager
      servableManager.load("/home/yansu/projects/test.yaml")
      val servable = servableManager.retriveModel("second-model", "1.0")
      val clusterServingServable = servable.asInstanceOf[ClusterServingServable]
      val instances = JsonUtil.fromJson(classOf[Instances], content)
      val inputs = instances.instances.map(instance => {
            InstancesPredictionInput(Instances(instance))
      })
      print(clusterServingServable.predict(inputs))
    }catch {
      case e =>
        println(e)
    }
  }

}