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
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim.SGD.{Poly, SequentialSchedule, Warmup}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.utils.{Engine, LoggerFilter, T, Table}
import com.intel.analytics.zoo.common.Optim.Fixed
import com.intel.analytics.zoo.common.{EveryEpoch, MaxEpoch, MaxIteration, SeveralIteration}
import com.intel.analytics.zoo.feature.pmem.{MemoryType, PARTITIONED}
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.EngineRef
import com.intel.analytics.zoo.pipeline.estimator.{ConstantClipping, Estimator, L2NormClipping}
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
        // When you are using incremental training, the training iteration may exceed the
        // polyIteration, if you are using MaxEpoch to end the training. So we add a very
        // small fixed learning rate to avoid a error thrown by SequentialSchedule.
        val lrSchedule = SequentialSchedule(iterationPerEpoch)
          .add(Warmup(warmupDelta), warmupIteration)
          .add(Poly(0.5, maxIteration), polyIteration)
          .add(Fixed(1e-10), Int.MaxValue)
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
        (EveryEpoch(), MaxEpoch(param.maxEpoch.get))
      } else {
        (SeveralIteration(param.checkpointIteration),
          MaxIteration(param.maxIteration))
      }
      if (param.gradientL2NormThreshold.isDefined) {
        estimator.setGradientClippingByL2Norm(param.gradientL2NormThreshold.get)
      } else if (param.gradientMin.isDefined && param.gradientMax.isDefined) {
        estimator.setConstantGradientClipping(param.gradientMin.get, param.gradientMax.get)
      }

      estimator.train(trainSet, ClassNLLCriterion[Float](),
        endTrigger = Some(endTrigger),
        checkPointTrigger = Some(checkpointTrigger),
        valSet, Array(new Top1Accuracy[Float], new Top5Accuracy[Float]))

      estimator.close()
      sc.stop()
    })
  }
}
