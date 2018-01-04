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
import com.intel.analytics.bigdl.pipeline.ssd.IOUtils
import com.intel.analytics.zoo.pipeline.common.nn.{MultiBoxLoss, MultiBoxLossParam}
import com.intel.analytics.zoo.pipeline.common.MeanAveragePrecision
import com.intel.analytics.zoo.pipeline.common.caffe.CaffeLoader
import com.intel.analytics.zoo.pipeline.ssd.model.SSDVgg
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.{Engine, LoggerFilter}
import com.intel.analytics.bigdl.visualization.{TrainSummary, ValidationSummary}
import com.intel.analytics.zoo.pipeline.common.dataset.roiimage.SSDMiniBatch
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import scopt.OptionParser

object TrainMessi {

  LoggerFilter.redirectSparkInfoLogs()
  Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.INFO)
  Logger.getLogger("com.intel.analytics.bigdl.pipeline").setLevel(Level.INFO)

  import Option._

  val logger = Logger.getLogger(getClass.getName)

  def main(args: Array[String]): Unit = {
    trainParser.parse(args, TrainParams()).map(param => {
      val conf = Engine.createSparkConf().setAppName(param.jobName)
        .set("spark.task.maxFailures", "1")
      val sc = new SparkContext(conf)
      Engine.init

      val trainSet = IOUtils.loadTrainSet(param.trainFolder, sc, param.resolution, param.batchSize)

      val valSet = IOUtils.loadValSet(param.valFolder, sc, param.resolution, param.batchSize)

      val model = if (param.modelSnapshot.isDefined) {
        Module.load[Float](param.modelSnapshot.get)
      } else {
        param.modelType match {
          case "vgg16" =>
            val model = SSDVgg(param.classNumber, param.resolution)
            if (param.weights.isDefined) {
              val m = Module.loadModule(param.weights.get)
              model.loadModelWeights(m)
            } else if (param.caffeDefPath.isDefined && param.caffeModelPath.isDefined) {
              CaffeLoader.load[Float](model,
                param.caffeDefPath.get, param.caffeModelPath.get, matchAll = false)
            }
            model
          case _ => throw new Exception("currently only test over vgg ssd model")
        }
      }

      val warmUpModel = if (param.warmUpMap.isDefined) {
        val optimMethod = new Adam[Float](
          learningRate = 0.0001,
          learningRateDecay = 0.0005
        )
        optimize(model, trainSet, valSet, param, optimMethod,
          Trigger.maxScore(param.warmUpMap.get.toFloat))
      } else {
        model
      }

      val optimMethod = if (param.stateSnapshot.isDefined) {
        OptimMethod.load[Float](param.stateSnapshot.get)
      } else {
        val learningRateSchedule = param.schedule match {
          case "multistep" =>
            val steps = if (param.learningRateSteps.isDefined) {
              param.learningRateSteps.get
            } else {
              Array[Int](80000 / 32 * param.batchSize, 100000 / 32 * param.batchSize,
                120000 / 32 * param.batchSize)
            }
            SGD.MultiStep(steps, param.learningRateDecay)
          case "plateau" =>
            SGD.Plateau(monitor = "score",
              factor = param.learningRateDecay.toFloat,
              patience = param.patience, minLr = 1e-5f, mode = "max")
        }
        new SGD[Float](
          learningRate = param.learningRate,
          momentum = 0.9,
          dampening = 0.0,
          learningRateSchedule = learningRateSchedule)
      }

      optimize(warmUpModel, trainSet, valSet, param, optimMethod,
        Trigger.maxEpoch(param.maxEpoch.get))

    })
  }

  private def optimize(model: Module[Float],
    trainSet: DataSet[SSDMiniBatch],
    valSet: DataSet[SSDMiniBatch], param: TrainParams, optimMethod: OptimMethod[Float],
    endTrigger: Trigger): Module[Float] = {
    val optimizer = Optimizer(
      model = model,
      dataset = trainSet,
      criterion = new MultiBoxLoss[Float](MultiBoxLossParam())
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
          nClass = param.classNumber)))
      .setEndWhen(endTrigger)
      .optimize()
  }
}
