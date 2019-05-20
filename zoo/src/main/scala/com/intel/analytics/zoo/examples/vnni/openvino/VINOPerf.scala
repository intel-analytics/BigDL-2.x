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

import java.util.Arrays

import com.intel.analytics.zoo.pipeline.inference.{InferenceModel, JTensor}
import org.apache.log4j.Logger
import scopt.OptionParser

import scala.util.Random


object VINOPerf {

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
      opt[Boolean]("isInt8")
        .text("Is Int8 optimized model?")
        .action((x, c) => c.copy(isInt8 = x))
    }

    parser.parse(args, ResNet50PerfParams()).foreach { param =>
      val batchSize = param.batchSize
      val iteration = param.iteration

      val randomData = Seq.fill(batchSize * 224 * 224 * 3)(Random.nextFloat())
        .toArray
      val input = new JTensor(randomData, Array(batchSize, 224, 224, 3))
      val batchInput = Arrays.asList(
        Arrays.asList({
          input
        }))

      val model = new InferenceModel(1)

      if (param.isInt8) {
        model.doLoadOpenVINOInt8(param.model, param.weight, param.batchSize)
      } else {
        model.doLoadOpenVINO(param.model, param.weight)
      }

      val predictStart = System.nanoTime()
      var averageLatency = 0L
      List.range(0, iteration).foreach { _ =>
        val start = System.nanoTime()
        if (param.isInt8) {
          model.doPredictInt8(batchInput)
        } else {
          model.doPredict(batchInput)
        }
        val latency = System.nanoTime() - start
        averageLatency += latency
        logger.info(s"Iteration $iteration latency is ${latency / 1e6} ms")
      }
      val totalTimeUsed = System.nanoTime() - predictStart
      val totalThroughput = "%.2f".format(batchSize * iteration / (totalTimeUsed / 1e9))
      logger.info(s"Average latency for iteration is ${averageLatency / iteration / 1e6} imgs/sec")
      logger.info(s"Takes $totalTimeUsed ns, throughput is $totalThroughput imgs/sec")
    }
  }
}
