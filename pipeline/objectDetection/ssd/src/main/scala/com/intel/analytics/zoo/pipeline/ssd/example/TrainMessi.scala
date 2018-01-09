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

package com.intel.analytics.zoo.pipeline.ssd.example

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.MiniBatch
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.optim.{Optimizer, _}
import com.intel.analytics.bigdl.pipeline.ssd.Utils
import com.intel.analytics.zoo.pipeline.common.nn.{MultiBoxLoss, MultiBoxLossParam}
import com.intel.analytics.zoo.pipeline.common.{MeanAveragePrecision, ModuleUtil}
import com.intel.analytics.zoo.pipeline.ssd.model.SSDVgg
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.{Engine, LoggerFilter}
import com.intel.analytics.bigdl.visualization.{TrainSummary, ValidationSummary}
import com.intel.analytics.zoo.pipeline.common.dataset.roiimage.SSDMiniBatch
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import scopt.OptionParser

import scala.io.Source

object TrainMessi {

  LoggerFilter.redirectSparkInfoLogs()
  Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.INFO)
  Logger.getLogger("com.intel.analytics.bigdl.pipeline").setLevel(Level.INFO)

  case class TrainParams(
    trainFolder: String = "./",
    valFolder: String = "./",
    resolution: Int = 300,
    checkpoint: Option[String] = None,
    modelSnapshot: Option[String] = None,
    stateSnapshot: Option[String] = None,
    className: String = "",
    batchSize: Int = -1,
    learningRate: Double = 0.001,
    overWriteCheckpoint: Boolean = false,
    maxEpoch: Int = 20,
    pretrain: Option[String] = None,
    jobName: String = "BigDL SSD Train Example",
    summaryDir: Option[String] = None
  )

  val trainParser = new OptionParser[TrainParams]("BigDL SSD Example") {
    opt[String]('f', "trainFolder")
      .text("url of hdfs folder store the train hadoop sequence files")
      .action((x, c) => c.copy(trainFolder = x))
    opt[String]('v', "valFolder")
      .text("url of hdfs folder store the validation hadoop sequence files")
      .action((x, c) => c.copy(valFolder = x))
    opt[Int]('r', "resolution")
      .text("input resolution 300 or 512")
      .action((x, c) => c.copy(resolution = x))
      .required()
    opt[String]("model")
      .text("model snapshot location")
      .action((x, c) => c.copy(modelSnapshot = Some(x)))
    opt[String]("weights")
      .text("pretrained weights")
      .action((x, c) => c.copy(pretrain = Some(x)))
    opt[String]("state")
      .text("state snapshot location")
      .action((x, c) => c.copy(stateSnapshot = Some(x)))
    opt[String]("checkpoint")
      .text("where to cache the model")
      .action((x, c) => c.copy(checkpoint = Some(x)))
    opt[Int]('e', "maxEpoch")
      .text("epoch numbers")
      .action((x, c) => c.copy(maxEpoch = x))
    opt[Double]('l', "learningRate")
      .text("inital learning rate")
      .action((x, c) => c.copy(learningRate = x))
      .required()
    opt[Int]('b', "batchSize")
      .text("batch size")
      .action((x, c) => c.copy(batchSize = x))
      .required()
    opt[String]("class")
      .text("class file")
      .action((x, c) => c.copy(className = x))
      .required()
    opt[Unit]("overWrite")
      .text("overwrite checkpoint files")
      .action((_, c) => c.copy(overWriteCheckpoint = true))
    opt[String]("name")
      .text("job name")
      .action((x, c) => c.copy(jobName = x))
    opt[String]("summary")
      .text("train validate summary")
      .action((x, c) => c.copy(summaryDir = Some(x)))
  }

  val logger = Logger.getLogger(getClass.getName)

  def main(args: Array[String]): Unit = {
    trainParser.parse(args, TrainParams()).map(param => {
      val conf = Engine.createSparkConf().setAppName(param.jobName)
        .set("spark.task.maxFailures", "1")
      val sc = new SparkContext(conf)
      Engine.init

      val classes = Source.fromFile(param.className).getLines().toArray
      val trainSet = Utils.loadTrainSet(param.trainFolder, sc, param.resolution, param.batchSize)

      val valSet = Utils.loadValSet(param.valFolder, sc, param.resolution, param.batchSize)

      val model = SSDVgg(classes.length, param.resolution)
      val m = Module.loadModule(param.pretrain.get)
      ModuleUtil.loadModelWeights(m, model, false)

      val optimMethod = new Adam[Float](
        learningRate = 0.0001,
        learningRateDecay = 0.0005
      )
      optimize(model, trainSet, valSet, param, optimMethod,
        Trigger.maxEpoch(param.maxEpoch), classes)

    })
  }

  private def optimize(model: Module[Float],
    trainSet: DataSet[SSDMiniBatch],
    valSet: DataSet[SSDMiniBatch], param: TrainParams, optimMethod: OptimMethod[Float],
    endTrigger: Trigger,
    classes: Array[String]): Module[Float] = {
    val optimizer = Optimizer(
      model = model,
      dataset = trainSet,
      criterion = new MultiBoxLoss[Float](MultiBoxLossParam(nClasses = classes.length))
    )

    if (param.checkpoint.isDefined) {
      optimizer.setCheckpoint(param.checkpoint.get, Trigger.everyEpoch)
    }

    if (param.overWriteCheckpoint) {
      optimizer.overWriteCheckpoint()
    }

    if (param.summaryDir.isDefined) {
      val trainSummary = TrainSummary(param.summaryDir.get, param.jobName)
      val validationSummary = ValidationSummary(param.summaryDir.get, param.jobName)
      trainSummary.setSummaryTrigger("LearningRate", Trigger.severalIteration(1))
      optimizer.setTrainSummary(trainSummary)
      optimizer.setValidationSummary(validationSummary)
    }
    optimizer
      .setOptimMethod(optimMethod)
      .setValidation(Trigger.everyEpoch,
        valSet.asInstanceOf[DataSet[MiniBatch[Float]]],
        Array(new MeanAveragePrecision(true, normalized = true, classes = classes)))
      .setEndWhen(endTrigger)
      .optimize()
  }
}
