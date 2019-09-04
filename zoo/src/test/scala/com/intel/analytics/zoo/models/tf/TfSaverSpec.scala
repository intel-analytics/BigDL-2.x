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

package com.intel.analytics.zoo.models.tf

import com.intel.analytics.bigdl.nn.Graph
import com.intel.analytics.bigdl.optim.L2Regularizer
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import com.intel.analytics.zoo.pipeline.api.keras.layers._
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.{FloatType, Tensor}
import com.intel.analytics.bigdl.utils.tf.TensorflowSaver
import com.intel.analytics.zoo.pipeline.api.Net

class TfSaverSpec extends ZooSpecHelper{
  "TfSaver" should "works" in {
    val modelGraph = Sequential().setName("model")
    // get 13 * 52 input matrix
    val reshape = Reshape(Array(13, 52), inputShape = Shape(676))
    // get 52 * 13 input matrix
    val transpose = Permute(Array(2, 1))
    val layer1 = LSTM[Float](outputDim = 100, returnSequences = true, wRegularizer = L2Regularizer(0.001))
    val layer2 = LSTM[Float](outputDim = 100, returnSequences = false, wRegularizer = L2Regularizer(0.001))
    val dropout = Dropout(0.5)
    val denseLayer = Dense(2, activation = "softmax", wRegularizer = L2Regularizer(0.001))
    modelGraph.add(reshape.setName("reshape"))
    modelGraph.add(transpose.setName("transpose"))
    modelGraph.add(layer1.setName("lstm1"))
    modelGraph.add(layer2.setName("lstm2"))
//    modelGraph.add(dropout.setName("dropout"))
    modelGraph.add(denseLayer.setName("output"))
    Net.saveToKeras2(modelGraph, "/tmp/my.h5", "/opt/anaconda3/envs/py36/bin/python")
    Net.saveToKeras2(modelGraph, "/tmp/my.h5")
//    val a = modelGraph.toModel().modules(0).asInstanceOf[Graph[Float]]
//    a.evaluate()


//    val b = com.intel.analytics.zoo.mo.tf.Tensorflow.placeholder(
//      FloatType, Array(123, 456).toSeq, "123")
//    com.intel.analytics.bigdl.utils.tf.Tensorflow.placeholder(FloatType, Array(123, 456).toSeq, "123")
//
//    TfSaver.saveGraph(a, Seq(("input", Seq(676))), "/tmp/tf123")
//    TensorflowSaver.saveGraph(a, Seq(("input", Seq(676))), "/tmp/tf123")

    val input = Tensor[Float].range(0, 676*2 - 1).resize(2, 676).div(1000)
    val o = modelGraph.forward(input)
    println(o)
    println()
  }


}
