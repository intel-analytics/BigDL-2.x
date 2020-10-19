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
import com.intel.analytics.bigdl.utils.{T, Table}
import com.intel.analytics.zoo.serving.PreProcessing
import com.intel.analytics.zoo.serving.arrow.{ArrowDeserializer, ArrowSerializer}
import com.intel.analytics.zoo.serving.engine.{ClusterServingInference, Timer}
import com.intel.analytics.zoo.serving.utils.{ClusterServingHelper, SerParams, Supportive}
import scopt.OptionParser

object InferenceBaseline extends Supportive {
  case class Params(configPath: String = "config.yaml",
                    testNum: Int = 1000,
                    parNum: Int = 1,
                    inputShape: String = "3, 224, 224",
                    singleMode: Boolean = false)
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
    opt[Boolean]("singleMode")
      .text("Use single mode to test")
      .action((x, params) => params.copy(singleMode = x))
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
  def getBase64StringOfTensor(activity: Activity): String = {
    val byteArr = ArrowSerializer.activityBatchToByte(activity, 1)
    Base64.getEncoder.encodeToString(byteArr)
  }
  def main(args: Array[String]): Unit = {
    val param = parser.parse(args, Params()).head
    val helper = new ClusterServingHelper()
    helper.initArgs()
    val sParam = new SerParams(helper)

    val model = helper.loadInferenceModel()
    val warmT = makeTensorFromShape(param.inputShape)
    val clusterServingInference = new ClusterServingInference(null, "openvino")
    clusterServingInference.typeCheck(warmT)
    clusterServingInference.dimCheck(warmT, "add", sParam.modelType)
    (0 until 10).foreach(_ => {
      val result = model.doPredict(warmT)
    })
    println("Warming up finished, begin baseline test...generating Base64 string")

    val b64string = getBase64StringOfTensor(warmT)
    println(s"Previewing base64 string, prefix is ${b64string.substring(0, 20)}")
    Thread.sleep(3000)


    val timer = new Timer()
    timing(s"Base line for single pipeline " +
      s"with input ${param.testNum.toString}") {
      var a = Seq[(String, Table)]()
      val pre = new PreProcessing(true)
      (0 until sParam.coreNum).foreach( i =>
        a = a :+ (i.toString(), T(warmT))
      )
      (0 until param.testNum).grouped(sParam.coreNum).flatMap(batch => {
          val t = timer.timing("Batch input", batch.size) {
            clusterServingInference.batchInput(a, sParam.coreNum, true, sParam.resize)
          }
          clusterServingInference.dimCheck(t, "add", sParam.modelType)
          val result = timer.timing("Inference", batch.size) {
            model.doPredict(t)
          }
          clusterServingInference.dimCheck(t, "remove", sParam.modelType)
          clusterServingInference.dimCheck(result, "remove", sParam.modelType)

          Seq()
      }).toArray
      timer.print()
    }




  }
}

