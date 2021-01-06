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
import com.intel.analytics.zoo.serving.PreProcessing
import com.intel.analytics.zoo.serving.arrow.{ArrowDeserializer, ArrowSerializer}
import com.intel.analytics.zoo.serving.engine.{ClusterServingInference, Timer}
import com.intel.analytics.zoo.serving.utils.{ClusterServingHelper, Supportive}
import scopt.OptionParser

object MockSinglePipelineBaseline extends Supportive {
  case class Params(configPath: String = "config.yaml",
                    testNum: Int = 1000,
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
  def getBase64StringOfTensor(activity: Activity): String = {
    val byteArr = ArrowSerializer.activityBatchToByte(activity, 1)
    Base64.getEncoder.encodeToString(byteArr)
  }
  def main(args: Array[String]): Unit = {
    val param = parser.parse(args, Params()).head
    val helper = new ClusterServingHelper()
    helper.loadConfig()

    val model = helper.loadInferenceModel()
    val warmT = makeTensorFromShape(param.inputShape)
    val clusterServingInference = new ClusterServingInference(null, "openvino")
    clusterServingInference.typeCheck(warmT)
    clusterServingInference.dimCheck(warmT, "add", helper.modelType)
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
      var a = Seq[(String, String)]()
      val pre = new PreProcessing(true)
      (0 until helper.coreNum).foreach( i =>
        a = a :+ (i.toString(), b64string)
      )
      (0 until param.testNum).grouped(helper.coreNum).flatMap(batch => {
        val preprocessed = timer.timing("Preprocess", batch.size) {
          (0 until batch.size).toParArray.map(i => {
            val deserializer = new ArrowDeserializer()
            val arr = deserializer.create(b64string)
            val tensor = Tensor(arr(0)._1, arr(0)._2)
            (a(i)._1, T(tensor))
          }).toIterator
        }
        preprocessed.grouped(batch.size).flatMap(b => {
          val t = timer.timing("Batch input", batch.size) {
            clusterServingInference.batchInput(b, helper.coreNum, true, helper.resize)
          }
          clusterServingInference.dimCheck(t, "add", helper.modelType)
          val result = timer.timing("Inference", batch.size) {
            model.doPredict(t)
          }
          clusterServingInference.dimCheck(t, "remove", helper.modelType)
          clusterServingInference.dimCheck(result, "remove", helper.modelType)
          val postprocessed = timer.timing("Postprocess", batch.size) {
            (0 until helper.coreNum).map(i => {
              ArrowSerializer.activityBatchToByte(result, i + 1)
            })
          }
          Seq(postprocessed)
        })

      }).toArray
      timer.print()
    }




  }
}

