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
package com.intel.analytics.zoo.examples.inception

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.models.inception.Inception_v1_NoAuxClassifier
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, Module}
import com.intel.analytics.bigdl.optim.SGD.{Poly, SequentialSchedule, Warmup}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.utils.{Engine, LoggerFilter, T, Table}
import com.intel.analytics.zoo.feature.pmem.MemoryType
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.EngineRef
import com.intel.analytics.zoo.pipeline.estimator.{Estimator}
import org.apache.spark.SparkContext

object TrainInceptionV1 {
  LoggerFilter.redirectSparkInfoLogs()

  import Options._

  def main(args: Array[String]): Unit = {
    trainParser.parse(args, new TrainParams()).map(param => {
      val imageSize = 224
      val conf = Engine.createSparkConf().setAppName("Analytics-zoo InceptionV1 Train Example")
        .set("spark.task.maxFailures", "1")
      val sc = new SparkContext(conf)
      Engine.init

      val trainSet = ImageNet2012(
        param.folder + "/train",
        sc,
        imageSize,
        param.batchSize,
        EngineRef.getNodeNumber(),
        EngineRef.getCoreNumber(),
        param.classNumber,
        MemoryType.fromString(param.memoryType),
        param.opencv
      )
      val valSet = ImageNet2012Val(
        param.folder + "/val",
        sc,
        imageSize,
        param.batchSize,
        EngineRef.getNodeNumber(),
        EngineRef.getCoreNumber(),
        param.classNumber,
        opencvPreprocessing = param.opencv
      )

      val model = if (param.modelSnapshot.isDefined) {
        Module.load[Float](param.modelSnapshot.get)
      } else if (param.graphModel) {
        Inception_v1_NoAuxClassifier.graph(classNum = param.classNumber)
      } else {
        Inception_v1_NoAuxClassifier(classNum = param.classNumber)
      }

      val iterationPerEpoch = math.ceil(1281167.toDouble / param.batchSize).toInt
      val maxIteration = if (param.maxEpoch.isDefined) {
        iterationPerEpoch * param.maxEpoch.get
      } else param.maxIteration

      val warmupIteration = param.warmupEpoch.getOrElse(0) * iterationPerEpoch

      val optimMethod = if (param.stateSnapshot.isDefined) {
        OptimMethod.load[Float](param.stateSnapshot.get)
      } else {
        val warmupDelta = if (warmupIteration == 0) 0.0
        else (param.maxLr.getOrElse(param.learningRate) - param.learningRate) / warmupIteration
        val polyIteration = maxIteration - warmupIteration
        val lrSchedule = SequentialSchedule(iterationPerEpoch)
          .add(Warmup(warmupDelta), warmupIteration).add(Poly(0.5, polyIteration), polyIteration)
        new SGD[Float](learningRate = param.learningRate, learningRateDecay = 0.0,
          weightDecay = param.weightDecay, momentum = 0.9, dampening = 0.0, nesterov = false,
          learningRateSchedule = lrSchedule)
      }
      val estimator = if (param.checkpoint.isDefined) {
        Estimator[Float](model, optimMethod, param.checkpoint.get)
      } else {
        Estimator[Float](model, optimMethod)
      }

      val (checkpointTrigger, endTrigger) = if (param.maxEpoch.isDefined) {
        (Trigger.everyEpoch, Trigger.maxEpoch(param.maxEpoch.get))
      } else {
        (Trigger.severalIteration(param.checkpointIteration),
          Trigger.maxIteration(param.maxIteration))
      }

      estimator.train(trainSet, ClassNLLCriterion[Float](),
        endTrigger = Some(endTrigger),
        checkPointTrigger = Some(checkpointTrigger),
        valSet, Array(new Top1Accuracy[Float], new Top5Accuracy[Float]))

      sc.stop()
    })
  }
}
