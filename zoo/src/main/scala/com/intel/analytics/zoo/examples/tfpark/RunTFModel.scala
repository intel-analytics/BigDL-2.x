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
      val userSize = 21037404
      val itemSize = 99
      val user = Tensor[Float](1000).apply1(_ => Random.nextInt(userSize))
      val item = Tensor[Float](1000).apply1(_ => Random.nextInt(itemSize))
      val label = Tensor[Float](1000).apply1(_ => Random.nextInt(1))

      val input = T(user, item, label)
      val model = TFTrainingHelper(param.modelPath)
      val criterion = new IdentityCriterion()
      val output = model.forward(input)
      print("fininsh forward")
      val loss = criterion.forward(output, output)
      val gradout = criterion.backward(output, output)
      model.backward(input, gradout)
      print("fininsh forward/backward")

    }
  }
}
