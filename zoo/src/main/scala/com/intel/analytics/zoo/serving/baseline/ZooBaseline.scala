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

import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.zoo.serving.engine.InferenceSupportive
import com.intel.analytics.zoo.serving.utils.{ClusterServingHelper, ConfigUtils, SerParams, Supportive}
import scopt.OptionParser

object ZooBaseline extends Supportive {
  case class Params(configPath: String = "config.yaml",
                    testNum: Int = 1000,
                    inputShape: String = "3, 224, 224")
  val parser = new OptionParser[Params]("Text Classification Example") {
    opt[String]('c', "configPath")
      .text("Config Path of Cluster Serving")
      .action((x, params) => params.copy(configPath = x))
    opt[Int]('n', "testNum")
      .text("Text Mode of Parallelism 1")
      .action((x, params) => params.copy(testNum = x))
    opt[String]('s', "inputShape")
      .text("Input Shape, split by coma")
      .action((x, params) => params.copy(inputShape = x))
  }
  def makeTensorFromShape(shapeStr: String): Activity = {
    val shapeArr = ConfigUtils.parseShape(shape = shapeStr)
    if (shapeArr.length == 1) {
      Tensor[Float](shapeArr(0)).rand()
    }
    else {
      throw new Error("multiple dim not supported yet")
    }
  }
  def main(args: Array[String]): Unit = {
    val param = parser.parse(args, Params()).head
    val helper = new ClusterServingHelper()
    helper.initArgs()
    val sParam = new SerParams(helper)

    val model = helper.loadInferenceModel()
    val warmT = makeTensorFromShape(param.inputShape)
    InferenceSupportive.typeCheck(warmT)
    InferenceSupportive.dimCheck(warmT, "add", sParam)
    (0 until 10).foreach(_ => {
      val result = model.doPredict(warmT)
    })
    print("Warming up finished, begin baseline test...")
    Thread.sleep(3000)

    timing(s"Baseline for input ${param.testNum.toString}") {
      if (sParam.inferenceMode == "single") {
        (0 until param.testNum).foreach(_ => {
          val t = makeTensorFromShape(param.inputShape)
          InferenceSupportive.typeCheck(t)
          InferenceSupportive.dimCheck(t, "add", sParam)
          val result = model.doPredict(t)
        })
      } else {
        var a = Seq[(String, Activity)]()
        (0 until sParam.coreNum).foreach( i =>
          a = a :+ (i.toString(), T(makeTensorFromShape(param.inputShape)))
        )
        (0 until param.testNum).grouped(sParam.coreNum).flatMap(i => {
          val t = timing("Batch input") {
            InferenceSupportive.batchInput(a, sParam)
          }
          InferenceSupportive.dimCheck(t, "add", sParam)
          val result = model.doPredict(t)
          Seq()
        }).toArray
      }
    }

  }
}
