package com.intel.analytics.zoo.serving

import com.intel.analytics.zoo.pipeline.api.net.TFNet
import com.intel.analytics.zoo.serving.http.ServingFrontendSerializer
import org.scalameter.{Key, Measurer, Warmer, config}
import scopt.OptionParser

import scala.collection.mutable.ArrayBuffer

object ArrayDien {
  case class Params(modelPath: String = "", dataPath: String = "")
  def main(args: Array[String]): Unit = {
    val parser = new OptionParser[Params]("DIEN") {
      opt[String]('m', "modelPath")
        .text("model path")
        .action((x, params) => params.copy(modelPath = x))
      opt[String]('d', "dataPath")
        .text("data Path")
        .action((x, params) => params.copy(dataPath = x))
    }
    val arg = parser.parse(args, Params()).head
    val timeArray = new ArrayBuffer[String]()
    (1 to 3).foreach(modelPar => {

      val modelArray = new Array[TFNet](3)
      (0 until 3).foreach(i => modelArray(i) = TFNet(arg.modelPath))

      (0 until 2).indices.toParArray.map(thrd => {
        val inputStr = scala.io.Source.fromFile(arg.dataPath).mkString
        val input = ServingFrontendSerializer.deserialize(inputStr)
        val time = config(
          Key.exec.benchRuns -> 100,
          Key.verbose -> true
        ) withWarmer {
          new Warmer.Default
        } withMeasurer {
          new Measurer.IgnoringGC
        } measure {
          modelArray(thrd).updateOutput(input)
        }
      })
      println(s"Total time of InferenceModel with concurrentNum $modelPar ended")
    })
  }
}

