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

package com.intel.analytics.zoo.examples.recommendation

import scopt.OptionParser

case class WNDParams(dataset: String = "ml-1m",
                     modelType: String = "wide_n_deep",
                     inputDir: String = "./data/ml-1m/",
                     batchSize: Int = 2048,
                     maxEpoch: Int = 10,
                     logDir: Option[String] = None)

object WideAndDeepExample {
  def main(args: Array[String]): Unit = {
    val defaultParams = WNDParams()
    val parser = new OptionParser[WNDParams]("WideAndDeep Example") {
      opt[String]("dataset")
        .text(s"dataset name, ml-1m or census")
        .required()
        .action((x, c) => c.copy(dataset = x))
      opt[String]("modelType")
        .text(s"modelType")
        .action((x, c) => c.copy(modelType = x))
      opt[String]("inputDir")
        .text(s"inputDir")
        .action((x, c) => c.copy(inputDir = x))
      opt[Int]('b', "batchSize")
        .text(s"batch size, default is 40")
        .action((x, c) => c.copy(batchSize = x))
      opt[Int]('e', "maxEpoch")
        .text(s"max epoch, default is 40")
        .action((x, c) => c.copy(maxEpoch = x))
      opt[String]("logDir")
        .text(s"logDir")
        .action((x, c) => c.copy(logDir = Some(x)))
    }
    parser.parse(args, defaultParams).map {
      params =>
        params.dataset match {
          case "ml-1m" => Ml1mWideAndDeep.run(params)
          case "census" => CensusWideAndDeep.run(params)
          case _ => throw new IllegalArgumentException(s"Unkown dataset name: ${params.dataset}." +
            s" Excepted ml-1m or census.")
        }
    } getOrElse {
      System.exit(1)
    }
  }


}
