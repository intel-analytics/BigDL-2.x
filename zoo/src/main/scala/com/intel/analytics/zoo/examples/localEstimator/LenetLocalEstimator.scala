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

package com.intel.analytics.zoo.examples.localEstimator

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.models.lenet.LeNet5
import com.intel.analytics.bigdl.nn.ClassNLLCriterion
import com.intel.analytics.bigdl.optim.{Adam, Loss, Top1Accuracy, Top5Accuracy}
import com.intel.analytics.zoo.pipeline.estimator.LocalEstimator
import org.slf4j.LoggerFactory

object LenetLocalEstimator extends App {

  val logger = LoggerFactory.getLogger(getClass)

  val imageDirPath = "hdfs://172.16.0.110:9000/mnist"

  val batchSize = 1000
  val epoch = 15
  val threadNum = 44

  val model: Module[Float] = LeNet5(10)
  val criterion = ClassNLLCriterion[Float]()
  val adam = new Adam[Float]()
  val validations = Array(new Top1Accuracy[Float], new Top5Accuracy[Float], new Loss[Float])
  val localEstimator = LocalEstimator(model, criterion, adam, validations, threadNum)
  logger.info(s"LocalEstimator loaded as $localEstimator")

  val trainData = MnistDataLoader.loadTrainData(imageDirPath)
  val testData = MnistDataLoader.loadTestData(imageDirPath)
  localEstimator.fit(trainData,
    testData,
    ImageProcessing.labeledGreyImageToMiniBatchTransformer,
    batchSize,
    epoch)

}
