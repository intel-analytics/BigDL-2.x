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

import com.intel.analytics.bigdl.nn.{Contiguous, CrossEntropyCriterion, Linear, Sequential, SpatialAveragePooling, Transpose, View}
import com.intel.analytics.bigdl.optim.{Adam, Loss, Top1Accuracy, Top5Accuracy}
import com.intel.analytics.zoo.pipeline.api.net.TFNet
import com.intel.analytics.zoo.pipeline.estimator.LocalEstimator

object TransferLearning extends App {
  val modelPath = "/root/glorysdj/arda-data/tf-ov-playground/working/resnet50-imagenet-fc2"
  val inputs = Array("resnet50_input:0")
  val outputs = Array("resnet50/activation_48/Relu:0")
  val originalModel = TFNet.fromSavedModel(modelPath, inputs, outputs)


  println(originalModel)

  val model = Sequential[Float]()
  model.add(Transpose[Float](Array((2, 4), (2, 3))))
  model.add(Contiguous[Float]())
  model.add(originalModel)
  model.add(Transpose[Float](Array((2, 4), (3, 4))))
  model.add(Contiguous[Float]())
  model.add(new SpatialAveragePooling[Float](2, 2, globalPooling = true))
  model.add(new View[Float](2048).setNumInputDims(3))
  model.add(new Linear[Float](2048, 2))

  println(model)

  val imageDirPath = "hdfs://172.16.0.110:9000/cifar10"
  val batchSize = 132
  val epoch = 20
  val threadNum = 10

  val criterion = new CrossEntropyCriterion[Float]()
  val adam = new Adam[Float]()
  val validations = Array(new Top1Accuracy[Float], new Loss[Float])
  val localEstimator = LocalEstimator(model, criterion, adam, validations, threadNum)
  println(s"LocalEstimator loaded as $localEstimator")

  // use only a little data for transfer learning
  val trainData = Cifar10DataLoader.loadTrainData(imageDirPath)
    .filter(_.label() <= 2).slice(0, 10 * batchSize)
  val testData = Cifar10DataLoader.loadTestData(imageDirPath)
    .filter(_.label() <= 2).slice(0, 10 * batchSize)

  localEstimator.fit(trainData,
    testData,
    ImageProcessing.labeledBGRImageToMiniBatchTransformer,
    batchSize,
    epoch)

}
