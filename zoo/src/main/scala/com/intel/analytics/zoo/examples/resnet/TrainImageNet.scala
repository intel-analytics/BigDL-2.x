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

package com.intel.analytics.zoo.examples.resnet

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.models.resnet.ResNet
import com.intel.analytics.bigdl.models.resnet.ResNet.{DatasetType, ShortcutType}
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.nn.mkldnn.ResNet.DatasetType.ImageNet
import com.intel.analytics.bigdl.nn.{BatchNormalization, Container, CrossEntropyCriterion, Module}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric._
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.examples.resnet.Utils._
import com.intel.analytics.zoo.feature.pmem.{DIRECT, DRAM, MemoryType, PMEM}
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.EngineRef
import com.intel.analytics.zoo.pipeline.estimator.Estimator
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf

object TrainImageNet {
  LoggerFilter.redirectSparkInfoLogs()
  Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.INFO)
  val logger = Logger.getLogger(getClass)

  def imageNetDecay(epoch: Int): Double = {
    if (epoch >= 80) {
      3
    } else if (epoch >= 60) {
      2
    } else if (epoch >= 30) {
      1
    } else {
      0.0
    }
  }



  def main(args: Array[String]): Unit = {
    trainParser.parse(args, new TrainParams()).map(param => {
      // initial zoo context
      val conf = new SparkConf().setAppName("resnet")
      val sc = NNContext.initNNContext(conf)

      val batchSize = param.batchSize
      val (imageSize, dataSetType, maxEpoch) =
        (224, DatasetType.ImageNet, param.nepochs)


      val trainDataSet = loadImageNetTrainDataSet(param.folder + "/train", sc, imageSize,
        batchSize, memoryType = MemoryType.fromString(param.memoryType))

      val validateDataSet = loadImageNetValDataSet(param.folder + "/val", sc, imageSize,
        batchSize)

      val shortcut: ShortcutType = ShortcutType.B

      val model = if (param.modelSnapshot.isDefined) {
        Module.load[Float](param.modelSnapshot.get)
      } else {
        EngineRef.getEngineType() match {
          case MklBlas =>
            val curModel =
              ResNet(classNum = param.classes, T("shortcutType" -> shortcut, "depth" -> param.depth,
                "optnet" -> param.optnet, "dataSet" -> dataSetType))
            if (param.optnet) {
              ResNet.shareGradInput(curModel)
            }
            ResNet.modelInit(curModel)

            /* Here we set parallism specificall for BatchNormalization and its Sub Layers, this is
            very useful especially when you want to leverage more computing resources like you want
            to use as many cores as possible but you cannot set batch size too big for each core due
            to the memory limitation, so you can set batch size per core smaller, but the smaller
            batch size will increase the instability of convergence, the synchronization among BN
            layers basically do the parameters synchronization among cores and thus will avoid the
            instability while improves the performance a lot. */
            val parallisim = EngineRef.getCoreNumber()
            setParallism(curModel, parallisim)

            curModel
          case MklDnn =>
            nn.mkldnn.ResNet.graph(param.batchSize / EngineRef.getNodeNumber(), param.classes,
              T("depth" -> 50, "dataSet" -> ImageNet))
        }
      }

      println(model)

      val optimMethod = if (param.stateSnapshot.isDefined) {
        val optim = OptimMethod.load[Float](param.stateSnapshot.get).asInstanceOf[SGD[Float]]
        val baseLr = param.learningRate
        val iterationsPerEpoch = math.ceil(1281167 / param.batchSize).toInt
        val warmUpIteration = iterationsPerEpoch * param.warmupEpoch
        val maxLr = param.maxLr
        val delta = (maxLr - baseLr) / warmUpIteration
        optim.learningRateSchedule = SGD.EpochDecayWithWarmUp(warmUpIteration, delta, imageNetDecay)
        optim
      } else {
        val baseLr = param.learningRate
        val iterationsPerEpoch = math.ceil(1281167 / param.batchSize).toInt
        val warmUpIteration = iterationsPerEpoch * param.warmupEpoch
        val maxLr = param.maxLr
        val delta = (maxLr - baseLr) / warmUpIteration

        logger.info(s"warmUpIteraion: $warmUpIteration, startLr: ${param.learningRate}, " +
          s"maxLr: $maxLr, " +
          s"delta: $delta, nesterov: ${param.nesterov}")
        new SGD[Float](learningRate = param.learningRate, learningRateDecay = 0.0,
          weightDecay = param.weightDecay, momentum = param.momentum, dampening = param.dampening,
          nesterov = param.nesterov,
          learningRateSchedule = SGD.EpochDecayWithWarmUp(warmUpIteration, delta, imageNetDecay))
      }

      val estimator = if (param.checkpoint.isDefined) {
        Estimator[Float](model, optimMethod, param.checkpoint.get)
      } else {
        Estimator[Float](model, optimMethod)
      }

      estimator.train(trainDataSet, new CrossEntropyCriterion[Float](),
        endTrigger = Some(Trigger.maxEpoch(param.nepochs)),
        checkPointTrigger = Some(Trigger.everyEpoch),
        validateDataSet, Array(new Top1Accuracy[Float], new Top5Accuracy[Float]))

      sc.stop()
    })
  }

  private def setParallism(model: AbstractModule[_, _, Float], parallism: Int): Unit = {
    if (model.isInstanceOf[BatchNormalization[Float]]) {
      model.asInstanceOf[BatchNormalization[Float]].setParallism(parallism)
    }
    if(model.isInstanceOf[Container[_, _, Float]]) {
      model.asInstanceOf[Container[_, _, Float]].
        modules.foreach(sub => setParallism(sub, parallism))
    }
  }
}
