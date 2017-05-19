package com.intel.analytics.util

import scopt.OptionParser


object parser {
  val parser = new OptionParser[LocalOptimizerPerfParam]("DeepSpeech2 Inference") {
    head("DS2 inference example")
    opt[String]('p', "path")
      .text("Path to save the model parameters")
      .action((v, p) => p.copy(path = v))
      .required()
    opt[String]("host")
      .text("The HDFS host:port")
      .action((v, p) => p.copy(host = v))
      .required()
    opt[Int]('b', "batchSize")
      .text("batch size")
      .action((v, p) => p.copy(batchSize = v))
    opt[Int]('n', "num")
      .text("file number")
      .action((v, p) => p.copy(num = v))
  }
}

case class LocalOptimizerPerfParam(
  host: String = null,
  path: String = null,
  batchSize: Int = 6,
  num: Int = 50)
