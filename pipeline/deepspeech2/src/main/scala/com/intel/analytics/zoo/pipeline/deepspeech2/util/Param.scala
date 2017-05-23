/*
 * Copyright 2016 The BigDL Authors.
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

package com.intel.analytics.zoo.pipeline.deepspeech2.util

import scopt.OptionParser

object parser {
  val parser = new OptionParser[LocalOptimizerPerfParam]("DeepSpeech2 Inference") {
    head("DS2 inference example")
    opt[String]('d', "dataPath")
      .text("data path (.wav or .flac files). Both local file and HDFS are accepted.")
      .action((v, p) => p.copy(dataPath = v))
      .required()
    opt[String]('m', "modelPath")
      .text("Path to load the model. For instance: /path/ds2.model")
      .action((v, p) => p.copy(modelPath = v))
      .required()
    opt[Int]('p', "partitionNum")
      .text("partition number. Default is 4.")
      .action((v, p) => p.copy(partition = v))
    opt[Int]('n', "num")
      .text("file number. Default is 8")
      .action((v, p) => p.copy(numFile = v))
    opt[Int]('s', "segment")
      .text("audio segment length in seconds. Default is 30")
      .action((v, p) => p.copy(segment = v))
  }
}

case class LocalOptimizerPerfParam(
  dataPath: String = null,
  modelPath: String = null,
  partition: Int = 4,
  numFile: Int = 8,
  segment: Int = 30)
