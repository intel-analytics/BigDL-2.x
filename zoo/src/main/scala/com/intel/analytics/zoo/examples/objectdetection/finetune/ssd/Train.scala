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

package com.intel.analytics.zoo.examples.objectdetection.finetune.ssd

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.dataset.MiniBatch
import com.intel.analytics.bigdl.optim.SGD._
import com.intel.analytics.bigdl.optim.{Optimizer, _}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.LoggerFilter
import com.intel.analytics.bigdl.visualization.{TrainSummary, ValidationSummary}
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.models.image.common.ImageModel
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.zoo.models.image.objectdetection.common.nn.{MultiBoxLoss, MultiBoxLossParam}
import com.intel.analytics.zoo.models.image.objectdetection.common.MeanAveragePrecision
import com.intel.analytics.zoo.models.image.objectdetection.common.dataset.roiimage.SSDMiniBatch
import com.intel.analytics.zoo.models.image.objectdetection.common.IOUtils
import org.apache.spark.SparkConf
import org.apache.log4j.{Level, Logger}
import scopt.OptionParser

import scala.io.Source

object Option {

  case class TrainParams(
    trainFolder: String = "./",
    valFolder: String = "./",
    resolution: Int = 300,
    checkpoint: Option[String] = None,
    modelSnapshot: Option[String] = None,
    stateSnapshot: Option[String] = None,
    className: String = "",
    batchSize: Int = 4,
    learningRate: Double = 0.001,
    schedule: String = "multistep",
    learningRateDecay: Double = 0.1,
    learningRateSteps: Option[Array[Int]] = None,
    patience: Int = 10,
    overWriteCheckpoint: Boolean = false,
    maxEpoch: Int = 5,
    jobName: String = "Analytics Zoo SSD Train Example",
    summaryDir: Option[String] = None,
    nPartition: Int = 1
  )

  val trainParser = new OptionParser[TrainParams]("Analytics Zoo SSD Example") {
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
      .required()
    opt[String]("state")
      .text("state snapshot location")
      .action((x, c) => c.copy(stateSnapshot = Some(x)))
    opt[String]("checkpoint")
      .text("where to cache the model")
      .action((x, c) => c.copy(checkpoint = Some(x)))
    opt[Int]("patience")
      .text("epoch to wait")
      .action((x, c) => c.copy(patience = x))
    opt[Int]('e', "maxEpoch")
      .text("epoch numbers")
      .action((x, c) => c.copy(maxEpoch = x))
      .required()
    opt[Double]('l', "learningRate")
      .text("inital learning rate")
      .action((x, c) => c.copy(learningRate = x))
    opt[String]("schedule")
      .text("learning rate schedule")
      .action((x, c) => c.copy(schedule = x))
    opt[Double]('d', "learningRateDecay")
      .text("learning rate decay")
      .action((x, c) => c.copy(learningRateDecay = x))
    opt[String]("step")
      .text("learning rate steps, split by ,")
      .action((x, c) => c.copy(learningRateSteps = Some(x.split(",").map(_.toInt))))
    opt[Int]('b', "batchSize")
      .text("batch size")
      .action((x, c) => c.copy(batchSize = x))
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
    opt[Int]('p', "partition")
      .text("number of partitions")
      .action((x, c) => c.copy(nPartition = x))
  }
}

object Train {
  LoggerFilter.redirectSparkInfoLogs()
  Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.INFO)
  Logger.getLogger("com.intel.analytics.zoo").setLevel(Level.INFO)

  import Option._

  val logger = Logger.getLogger(getClass.getName)

  def main(args: Array[String]): Unit = {
    trainParser.parse(args, TrainParams()).map(param => {
      val conf = new SparkConf().setAppName(param.jobName)
      val sc = NNContext.initNNContext(conf)

      val classes = Source.fromFile(param.className).getLines().toArray
      val trainSet = IOUtils.loadSSDTrainSet(param.trainFolder, sc, param.resolution, param.batchSize,
        param.nPartition)

      val valSet = IOUtils.loadSSDValSet(param.valFolder, sc, param.resolution, param.batchSize,
        param.nPartition)

//      val model = ImageModel.loadModel[Float](param.modelSnapshot.get)
      val model = Module.loadModule[Float]("/home/ding/proj/analytics-zoo-debug/bigdl_ssd-mobilenet-300x300_PASCAL_0.4.0.model")
//val test1 = Tensor[Float](4, 3, 300, 300).rand
//      val test2 = Tensor[Float](1, 2, 7668)
//      model.forward(test1)
//      model.backward(test1, test2)

      val optimMethod = if (param.stateSnapshot.isDefined) {
        OptimMethod.load[Float](param.stateSnapshot.get)
      } else {
        val learningRateSchedule = param.schedule match {
          case "multistep" =>
            val steps = if (param.learningRateSteps.isDefined) {
              param.learningRateSteps.get
            } else {
              Array[Int](80000 * 32 / param.batchSize, 100000 * 32 / param.batchSize,
                120000 * 32 / param.batchSize)
            }
            val lrSchedules = new SequentialSchedule(74)
            val delta = (param.learningRate - 0.001) / 370
            lrSchedules.add(Warmup(delta), 370).add(SGD.MultiStep(steps, param.learningRateDecay),
              param.maxEpoch * 74)
            lrSchedules
          case "plateau" =>
            val lrSchedules = new SequentialSchedule(74)
            val delta = (param.learningRate - 0.001) / 370
            lrSchedules.add(Warmup(delta), 370).add(SGD.Plateau(monitor = "score",
              factor = param.learningRateDecay.toFloat,
              patience = param.patience, minLr = 1e-5f, mode = "max"), param.maxEpoch * 74)
            lrSchedules
        }
        new SGD[Float](
          learningRate = param.learningRate,
          momentum = 0.9,
          dampening = 0.0,
          learningRateSchedule = learningRateSchedule)
      }

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
        Array(new MeanAveragePrecision(true, normalized = true,
          classes = classes)))
      .setEndWhen(endTrigger)
      .optimize()
  }
}
