package com.intel.analytics.zoo.examples.tfpark


import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.zoo.tfpark.{IdentityCriterion, TFTrainingHelper}
import scopt.OptionParser

import scala.util.Random

case class TFModelParams(
                          dataPath: String = "./",
                          modelPath: String = "./")


object RunTFModel {

  def main(args: Array[String]): Unit = {
    val parser = new OptionParser[TFModelParams]("TextClassification Example") {
      opt[String]("dataPath")
        .text("The directory containing the training data")
        .action((x, c) => c.copy(dataPath = x))
      opt[String]("modelPath")
        .required()
        .text("model path")
        .action((x, c) => c.copy(modelPath = x))
    }

    parser.parse(args, TFModelParams()).map { param =>
//      val userSize = 21037404
      val userSize = 100
      val itemSize = 99
      val user = Tensor[Float](1000).apply1(_ => Random.nextInt(userSize))
      val item = Tensor[Float](1000).apply1(_ => Random.nextInt(itemSize))
      val label = Tensor[Float](1000).apply1(_ => Random.nextInt(1))

      val input = T(user, item, label)
      val model = TFTrainingHelper(param.modelPath)
      val graphDef = model.graphRunner.graphDef
      println(s"graph def size is: ${graphDef.length} bytes")
      val (weights, gradweights) = model.parameters()
      for (i <- 0 until weights.length -1 ) {
        println(s"weights ${i} size: ${weights(i).size().mkString(",")}")
      }
//      for (i <- 0 until gradweights.length -1 ){
//        println(s"gradweights ${i} size: ${gradweights(i).size().mkString(",")}")
//      }
      val extraParams = model.getExtraParameter()
      for (i <- 0 until extraParams.length -1 ){
        println(s"extraParams ${i} size: ${extraParams(i).size().mkString(",")}")
      }

//      val graphOut = model.getGraphOut()
//      for (i <- 0 until graphOut.length -1 ){
//        println(s"graphOut ${i} size: ${graphOut(i).size().mkString(",")}")
//      }

      val criterion = new IdentityCriterion()
      val output = model.forward(input)
      println("fininsh forward")
      val loss = criterion.forward(output, output)
      val gradout = criterion.backward(output, output)
      model.backward(input, gradout)
      println("fininsh forward/backward")

//      val (weights, gradweights) = model.parameters()
//      for (i <- 0 until weights.length -1 ) {
//        println(s"weights ${i} size: ${weights(i).size().mkString(",")}")
//      }
//      for (i <- 0 until gradweights.length -1 ){
//        println(s"gradweights ${i} size: ${gradweights(i).size().mkString(",")}")
//      }
//      val extraParams = model.getExtraParameter()
//      for (i <- 0 until extraParams.length -1 ){
//        println(s"extraParams ${i} size: ${extraParams(i).size().mkString(",")}")
//      }
//
//      val graphOut = model.getGraphOut()
//      for (i <- 0 until graphOut.length -1 ){
//        println(s"graphOut ${i} size: ${graphOut(i).size().mkString(",")}")
//      }


      val output2 = model.forward(input)
      println("fininsh forward")
      val loss2 = criterion.forward(output2, output2)
      val gradout2 = criterion.backward(output2, output2)
      model.backward(input, gradout2)
      println("fininsh forward/backward")

    }
  }
}
