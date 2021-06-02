/*
 * Copyright 2018 Analytics Zoo Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


package com.intel.analytics.zoo.serving.baseline


import java.util.Base64

import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.zoo.serving.ClusterServing.helper
import com.intel.analytics.zoo.serving.{ClusterServing, PreProcessing}
import com.intel.analytics.zoo.serving.serialization.{ArrowDeserializer, ArrowSerializer}
import com.intel.analytics.zoo.serving.engine.{ClusterServingInference, Timer}
import com.intel.analytics.zoo.serving.utils.{ClusterServingHelper, ConfigParser, Conventions, Supportive}
import org.apache.log4j.{Level, Logger}
import scopt.OptionParser

object MockSingleThreadInferenceMultiplePipeline extends Supportive {
  case class Params(configPath: String = "config.yaml",
                    testNum: Int = 500,
                    parNum: Int = 1,
                    inputShape: String = "3, 224, 224")
  val parser = new OptionParser[Params]("Text Classification Example") {
    opt[String]('c', "configPath")
      .text("Config Path of Cluster Serving")
      .action((x, params) => params.copy(configPath = x))
    opt[Int]('n', "testNum")
      .text("Number of test input")
      .action((x, params) => params.copy(testNum = x))
    opt[Int]('p', "parallelism")
      .text("Parallelism number, align to Flink -p")
      .action((x, params) => params.copy(parNum = x))
    opt[String]('s', "inputShape")
      .text("Input Shape, split by coma")
      .action((x, params) => params.copy(inputShape = x))
  }
  def parseShape(shape: String): Array[Array[Int]] = {
    val shapeListStr = shape.
      split("""\[\[|\]\]|\],\s*\[""").filter(x => x != "")
    var shapeList = new Array[Array[Int]](shapeListStr.length)
    (0 until shapeListStr.length).foreach(idx => {
      val arr = shapeListStr(idx).stripPrefix("[").stripSuffix("]").split(",")
      val thisShape = new Array[Int](arr.length)
      (0 until arr.length).foreach(i => {
        thisShape(i) = arr(i).trim.toInt
      })
      shapeList(idx) = thisShape
    })
    shapeList
  }
  def makeTensorFromShape(shapeStr: String): Activity = {
    val shapeArr = parseShape(shape = shapeStr)
    if (shapeArr.length == 1) {
      Tensor[Float](shapeArr(0)).rand()
    }
    else {
      throw new Error("multiple dim not supported yet")
    }
  }
  def getBase64StringOfTensor(): String = {
    val resource = getClass.getClassLoader.getResource("serving")

    val dataPath = resource.getPath + "/image-3_224_224-base64"
    scala.io.Source.fromFile(dataPath).mkString
  }
  def main(args: Array[String]): Unit = {
    Logger.getLogger("com.intel.analytics.zoo").setLevel(Level.DEBUG)
    val param = parser.parse(args, Params()).head
    val configParser = new ConfigParser(param.configPath)
    helper = configParser.loadConfig()

    ClusterServing.model = helper.loadInferenceModel(param.parNum)
    val warmT = makeTensorFromShape(param.inputShape)


    val timer = new Timer()
    for (thrdId <- 0 to 4) {
      val clusterServingInference = new ClusterServingInference()
      clusterServingInference.typeCheck(warmT)
      clusterServingInference.dimCheck(warmT, "add", helper.modelType)
      (0 until 10).foreach(_ => {
        val result = ClusterServing.model.doPredict(warmT)
      })
      println(s"Thread $thrdId Warming up finished, begin baseline test...generating Base64 string")

      val b64string = getBase64StringOfTensor()
      println(s"Previewing base64 string, prefix is ${b64string.substring(0, 20)}")
      Thread.sleep(1000)

      timing(s"Thread $thrdId Baseline for SingleThreadInferenceMultiplePipeline " +
        s"with input ${param.testNum.toString}") {
        var a = Seq[(String, String, String)]()
        val pre = new PreProcessing()
        pre.helper.chwFlag = true
        (0 until helper.threadPerModel).foreach(i =>
          a = a :+ (i.toString(), b64string, "")
        )
        (0 until param.testNum).map(_ => {
          clusterServingInference.singleThreadPipeline(a.toList)
        }).toArray

      }
    }





  }
}

