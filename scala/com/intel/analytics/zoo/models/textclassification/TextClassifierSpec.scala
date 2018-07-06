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

package com.intel.analytics.zoo.models.textclassification

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.models.common.ZooModel
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest
import org.apache.spark.{SparkConf, SparkContext}

class TextClassifierSpec extends ZooSpecHelper {

  private var sc: SparkContext = _

  override def doBefore(): Unit = {
    val conf = new SparkConf()
      .setMaster("local[4]")
    sc = NNContext.initNNContext(conf, appName = "TrainingSpec")
  }

  override def doAfter(): Unit = {
    if (sc != null) {
      sc.stop()
    }
  }

  "TextClassifier model" should "compute the correct output shape" in {
    val model = TextClassifier[Float](classNum = 20, tokenLength = 200).buildModel()
    model.getOutputShape().toSingle().toArray should be (Array(-1, 20))
  }

  "TextClassifier lstm forward and backward" should "work properly" in {
    val model = TextClassifier[Float](classNum = 15, tokenLength = 10, sequenceLength = 20,
      encoder = "lstm")
    model.summary()
    val input = Tensor[Float](Array(1, 20, 10)).rand()
    val output = model.forward(input)
    val gradInput = model.backward(input, output)
  }

  "TextClassifier gru forward and backward" should "work properly" in {
    val model = TextClassifier[Float](classNum = 5, tokenLength = 15, sequenceLength = 40,
      encoder = "gru")
    val input = Tensor[Float](Array(1, 40, 15)).rand()
    val output = model.forward(input)
    val gradInput = model.backward(input, output)
  }

  "TextClassifier predictClasses giving zero-based label" should "work properly" in {
    val model = TextClassifier[Float](classNum = 10, tokenLength = 20)
    val data = new Array[Sample[Float]](10)
    var i = 0
    while (i < data.length) {
      val input = Tensor[Float](500, 20).rand()
      val label = Tensor[Float](1).fill(0.0f)
      data(i) = Sample(input, label)
      i += 1
    }
    val dataSet = sc.parallelize(data, 2)
    val result = model.predictClasses(dataSet)

    val prob = result.collect()
    prob.zip(data).foreach(item => {
      val res = model.forward(item._2.feature.reshape(Array(1, 500, 20)))
        .toTensor[Float].squeeze().max(1)._2.valueAt(1).toInt
      (res-1) should be (item._1)
    })
  }

}

class TextClassifierSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val model = TextClassifier[Float](classNum = 20, tokenLength = 50, sequenceLength = 100)
    val input = Tensor[Float](Array(1, 100, 50)).rand()
    ZooSpecHelper.testZooModelLoadSave(
      model.asInstanceOf[ZooModel[Tensor[Float], Tensor[Float], Float]],
      input, TextClassifier.loadModel[Float])
  }
}
