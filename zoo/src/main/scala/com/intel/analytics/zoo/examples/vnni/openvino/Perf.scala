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

package com.intel.analytics.zoo.examples.vnni.openvino

import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.zoo.pipeline.inference.InferenceModel
import org.apache.log4j.Logger
import scopt.OptionParser


case class ResNet50PerfParams(model: String = "",
                              weight: String = "",
                              batchSize: Int = 4,
                              iteration: Int = 1000)

object Perf {

  val logger: Logger = Logger.getLogger(getClass)

  def main(args: Array[String]): Unit = {

    val parser = new OptionParser[ResNet50PerfParams]("ResNet50 Int8 Performance Test") {
      opt[String]('m', "model")
        .text("The path to the int8 quantized ResNet50 model snapshot")
        .action((v, p) => p.copy(model = v))
        .required()
      opt[String]('w', "weight")
        .text("The path to the int8 ResNet50 model weight")
        .action((v, p) => p.copy(weight = v))
      opt[Int]('b', "batchSize")
        .text("Batch size of input data")
        .action((v, p) => p.copy(batchSize = v))
      opt[Int]('i', "iteration")
        .text("Iteration of perf test. The result will be average of each iteration time cost")
        .action((v, p) => p.copy(iteration = v))
    }

    parser.parse(args, ResNet50PerfParams()).foreach { param =>
      val batchSize = param.batchSize
      val batchInput = Tensor(Array(batchSize, 3, 224, 224)).rand().addSingletonDimension()
//      val singleInput = Tensor(Array(1, 3, 224, 224)).rand()
      Engine.init

      val model = new InferenceModel(4)
      model.doLoadOpenVINO(param.model, param.weight)

      var iteration = 0
      while (iteration < param.iteration) {
        val start = System.nanoTime()
        model.doPredict(batchInput)
        val timeUsed = System.nanoTime() - start
        val throughput = "%.2f".format(batchSize.toFloat / (timeUsed / 1e9))
        logger.info(s"Iteration $iteration, takes $timeUsed ns, throughput is $throughput imgs/sec")

        iteration += 1
      }

    }
  }
}