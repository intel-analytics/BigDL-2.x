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
import com.intel.analytics.bigdl.nn.{Module, SpatialShareConvolution}
import com.intel.analytics.bigdl.optim.{Optimizer, _}
import com.intel.analytics.bigdl.pipeline.fasterrcnn.Utils
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.{Engine, LoggerFilter}
import com.intel.analytics.bigdl.visualization.{TrainSummary, ValidationSummary}
import com.intel.analytics.zoo.pipeline.common.dataset.{FrcnnMiniBatch, PascalVoc}
import com.intel.analytics.zoo.pipeline.common.{IOUtils, MeanAveragePrecision}
import com.intel.analytics.zoo.pipeline.common.nn.FrcnnCriterion
import com.intel.analytics.zoo.pipeline.fasterrcnn.model.{VggFRcnn, _}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import scopt.OptionParser

object Option {

  case class TrainParams(
    trainFolder: String = "./",
    valFolder: String = "./",
    modelType: String = "vgg16",
    pretrain: String = "",
    caffeModelPath: Option[String] = None,
    optim: String = "sgd",
    checkpoint: Option[String] = None,
    modelSnapshot: Option[String] = None,
    stateSnapshot: Option[String] = None,
    classNumber: Int = 21,
    batchSize: Int = -1,
    learningRate: Double = 0.001,
    step: Int = 50000,
    maxIter: Int = 50,
    weights: Option[String] = None,
    jobName: String = "BigDL SSD Train Example",
    summaryDir: Option[String] = None,
    checkIter: Int = 200,
    share: Boolean = true
  )

  val trainParser = new OptionParser[TrainParams]("BigDL SSD Example") {
    opt[String]('f', "trainFolder")
      .text("url of hdfs folder store the train hadoop sequence files")
      .action((x, c) => c.copy(trainFolder = x))
    opt[String]('v', "valFolder")
      .text("url of hdfs folder store the validation hadoop sequence files")
      .action((x, c) => c.copy(valFolder = x))
    opt[String]('t', "modelType")
      .text("net type : vgg16")
      .action((x, c) => c.copy(modelType = x))
      .required()
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
    opt[Int]("classNum")
      .text("class number")
      .action((x, c) => c.copy(classNumber = x))
    opt[Int]("checkIter")
      .text("checkpoint iteration")
      .action((x, c) => c.copy(checkIter = x))
    opt[String]("name")
      .text("job name")
      .action((x, c) => c.copy(jobName = x))
    opt[String]("summary")
      .text("train validate summary")
      .action((x, c) => c.copy(summaryDir = Some(x)))
    opt[Boolean]("share")
      .text("share convolution")
      .action((x, c) => c.copy(share = x))
  }
}


object Train {

  LoggerFilter.redirectSparkInfoLogs()
  Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.INFO)
  Logger.getLogger("com.intel.analytics.bigdl.pipeline").setLevel(Level.INFO)

  import Option._

  val logger = Logger.getLogger(getClass.getName)

  def main(args: Array[String]): Unit = {
    trainParser.parse(args, TrainParams()).map(param => {
      val conf = Engine.createSparkConf().setAppName(param.jobName)
      val sc = new SparkContext(conf)
      Engine.init

      var (model, preParamTrain, preParamVal, postParam) = param.modelType match {
        case "vgg16" =>
          val postParam = PostProcessParam(0.3f, param.classNumber, false, 100, 0.05)
          val preParamTrain = PreProcessParam(param.batchSize, Array(400, 500, 600, 700))
          val preParamVal = PreProcessParam(param.batchSize, nPartition = param.batchSize)
          val model = if (param.modelSnapshot.isDefined) {
            Module.load[Float](param.modelSnapshot.get)
          } else {
            val pretrain = Module.loadModule[Float](param.pretrain)
            val model = VggFRcnn(param.classNumber, postParam)
            model.loadModelWeights(pretrain, false)
          }
          (model, preParamTrain, preParamVal, postParam)
        case "pvanet" =>
          val postParam = PostProcessParam(0.4f, param.classNumber, true, 100, 0.05)
          val preParamTrain = PreProcessParam(param.batchSize, Array(640), 32)
          val preParamVal = PreProcessParam(param.batchSize, Array(640), 32)
//          val model = Module.loadCaffe(PvanetFRcnn(param.classNumber, postParam),
//            param.caffeDefPath.get, param.caffeModelPath.get)
          val model = if (param.modelSnapshot.isDefined) {
            Module.load[Float](param.modelSnapshot.get)
          } else {
            val pretrain = Module.loadModule[Float](param.pretrain)
            val model = PvanetFRcnn(param.classNumber, postParam)
            model.loadModelWeights(pretrain, false)
          }
          (model, preParamTrain, preParamVal, postParam)
        case _ =>
          throw new Exception("unsupport network")
      }
      model = if (param.share) SpatialShareConvolution.shareConvolution(model) else model

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

      optimize(model, trainSet, valSet, param, optimMethod,
        Trigger.maxIteration(param.maxIter), new FrcnnCriterion())

    })
  }

  private def optimize(model: Module[Float],
    trainSet: DataSet[FrcnnMiniBatch],
    valSet: DataSet[FrcnnMiniBatch], param: TrainParams, optimMethod: OptimMethod[Float],
    endTrigger: Trigger,
    criterion: Criterion[Float]): Module[Float] = {
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
        Array(new MeanAveragePrecision(use07metric = true, normalized = false,
          // todo: update it with user-defined
          classes = PascalVoc.classes)))
      .setEndWhen(endTrigger)
      .optimize()
  }
}
