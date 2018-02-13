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
 *
 */

package com.intel.analytics.zoo.pipeline.fasterrcnn.example

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.MiniBatch
import com.intel.analytics.bigdl.nn.{Module}
import com.intel.analytics.bigdl.optim.{Optimizer, _}
import com.intel.analytics.bigdl.pipeline.fasterrcnn.Utils
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.{Engine, LoggerFilter}
import com.intel.analytics.bigdl.visualization.{TrainSummary, ValidationSummary}
import com.intel.analytics.zoo.pipeline.common.dataset.{FrcnnMiniBatch}
import com.intel.analytics.zoo.pipeline.common.{MeanAveragePrecision}
import com.intel.analytics.zoo.pipeline.common.nn.FrcnnCriterion
import com.intel.analytics.zoo.pipeline.fasterrcnn.model.{VggFRcnn, _}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import scopt.OptionParser

import scala.io.Source

object Option {

  case class TrainParams(
    trainFolder: String = "./",
    valFolder: String = "./",
    pretrain: String = "",
    optim: String = "sgd",
    checkpoint: Option[String] = None,
    modelSnapshot: Option[String] = None,
    stateSnapshot: Option[String] = None,
    className: String = "",
    batchSize: Int = -1,
    learningRate: Double = 0.001,
    step: Int = 50000,
    maxIter: Int = 50,
    weights: Option[String] = None,
    jobName: String = "BigDL SSD Train Example",
    summaryDir: Option[String] = None,
    checkIter: Int = 200
  )

  val trainParser = new OptionParser[TrainParams]("BigDL SSD Example") {
    opt[String]('f', "trainFolder")
      .text("url of hdfs folder store the train hadoop sequence files")
      .action((x, c) => c.copy(trainFolder = x))
    opt[String]('v', "valFolder")
      .text("url of hdfs folder store the validation hadoop sequence files")
      .action((x, c) => c.copy(valFolder = x))
    opt[String]("pretrain")
      .text("pretrained imagenet model")
      .action((x, c) => c.copy(pretrain = x))
    opt[String]("model")
      .text("model snapshot location")
      .action((x, c) => c.copy(modelSnapshot = Some(x)))
    opt[String]("weights")
      .text("pretrained weights")
      .action((x, c) => c.copy(weights = Some(x)))
    opt[String]("state")
      .text("state snapshot location")
      .action((x, c) => c.copy(stateSnapshot = Some(x)))
    opt[String]("checkpoint")
      .text("where to cache the model")
      .action((x, c) => c.copy(checkpoint = Some(x)))
    opt[Int]("step")
      .text("step to decay learning rate")
      .action((x, c) => c.copy(step = x))
    opt[Int]('i', "maxIter")
      .text("iteration numbers")
      .action((x, c) => c.copy(maxIter = x))
    opt[Double]('l', "learningRate")
      .text("inital learning rate")
      .action((x, c) => c.copy(learningRate = x))
      .required()
    opt[String]("optim")
      .text("optim method")
      .action((x, c) => c.copy(optim = x))
    opt[Int]('b', "batchSize")
      .text("batch size")
      .action((x, c) => c.copy(batchSize = x))
      .required()
    opt[String]("class")
      .text("class names")
      .action((x, c) => c.copy(className = x))
    opt[Int]("checkIter")
      .text("checkpoint iteration")
      .action((x, c) => c.copy(checkIter = x))
    opt[String]("name")
      .text("job name")
      .action((x, c) => c.copy(jobName = x))
    opt[String]("summary")
      .text("train validate summary")
      .action((x, c) => c.copy(summaryDir = Some(x)))
  }
}


object Train {

  LoggerFilter.redirectSparkInfoLogs()
  Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.INFO)
  Logger.getLogger("com.intel.analytics.zoo.pipeline").setLevel(Level.INFO)

  import Option._

  val logger = Logger.getLogger(getClass.getName)

  def main(args: Array[String]): Unit = {
    trainParser.parse(args, TrainParams()).map(param => {
      val conf = Engine.createSparkConf().setAppName(param.jobName)
      val sc = new SparkContext(conf)
      val classNames = Source.fromFile(param.className).getLines().toArray
      Engine.init
      val postParam = PostProcessParam(0.3f, classNames.length, false, 100, 0.05)
      val preParamTrain = PreProcessParam(param.batchSize, Array(400, 500, 600, 700))
      val preParamVal = PreProcessParam(param.batchSize, nPartition = param.batchSize)
      val model = if (param.modelSnapshot.isDefined) {
        Module.load[Float](param.modelSnapshot.get)
      } else {
        val pretrain = Module.loadModule[Float](param.pretrain)
        val model = VggFRcnn(classNames.length, postParam)
        model.loadModelWeights(pretrain, false)
      }

      val trainSet = Utils.loadTrainSet(param.trainFolder, sc, preParamTrain, param.batchSize)

      val valSet = Utils.loadValSet(param.valFolder, sc, preParamVal, param.batchSize)

      val optimMethod = if (param.stateSnapshot.isDefined) {
        OptimMethod.load[Float](param.stateSnapshot.get)
      } else {
        param.optim match {
          case "sgd" =>
            val learningRateSchedule = SGD.Step(param.step, 0.1)
            new SGD[Float](
              learningRate = param.learningRate,
              momentum = 0.9,
              dampening = 0.0,
              learningRateSchedule = learningRateSchedule,
              weightDecay = 0.0005)
          case "adam" =>
            new Adam[Float](
              learningRate = param.learningRate
            )
        }
      }

      val meanAveragePrecision = new MeanAveragePrecision(use07metric = true, normalized = false,
        classes = classNames)
      optimize(model, trainSet, valSet, param, optimMethod,
        Trigger.maxIteration(param.maxIter), new FrcnnCriterion(), meanAveragePrecision)

    })
  }

  private def optimize(model: Module[Float],
    trainSet: DataSet[FrcnnMiniBatch],
    valSet: DataSet[FrcnnMiniBatch], param: TrainParams, optimMethod: OptimMethod[Float],
    endTrigger: Trigger,
    criterion: Criterion[Float],
    validationMethod: ValidationMethod[Float]): Module[Float] = {
    val optimizer = Optimizer(
      model = model,
      dataset = trainSet,
      criterion = criterion
    )

    if (param.checkpoint.isDefined) {
      optimizer.setCheckpoint(param.checkpoint.get, Trigger.severalIteration(param.checkIter))
    }

    optimizer.overWriteCheckpoint()

    if (param.summaryDir.isDefined) {
      val trainSummary = TrainSummary(param.summaryDir.get, param.jobName)
      val validationSummary = ValidationSummary(param.summaryDir.get, param.jobName)
      trainSummary.setSummaryTrigger("LearningRate", Trigger.severalIteration(1))
      optimizer.setTrainSummary(trainSummary)
      optimizer.setValidationSummary(validationSummary)
    }
    optimizer
      .setOptimMethod(optimMethod)
      .setValidation(Trigger.severalIteration(param.checkIter),
        valSet.asInstanceOf[DataSet[MiniBatch[Float]]],
        Array(validationMethod))
      .setEndWhen(endTrigger)
      .optimize()
  }
}
