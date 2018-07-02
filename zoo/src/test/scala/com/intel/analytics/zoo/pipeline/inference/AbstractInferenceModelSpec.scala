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
package com.intel.analytics.zoo.pipeline.inference

import java.{lang, util}

import com.intel.analytics.bigdl.tensor.Tensor
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class AbstractInferenceModelSpec extends FlatSpec with Matchers with BeforeAndAfter {

  "AbstractInferenceModel " should "works with TFNet" in {
    val model = new AbstractInferenceModel() {}
    val resource = getClass().getClassLoader().getResource("tfnet")
    model.loadTF(resource.getPath)
    val arr = new util.ArrayList[lang.Float]()
    (0 until 28 * 28).foreach(_ => arr.add(math.random.toFloat))
    val shape = new util.ArrayList[Integer]()
    shape.add(28)
    shape.add(28)
    shape.add(1)
    val inputTensor = new JTensor(arr, shape)
    val input = new util.ArrayList[JTensor]()
    input.add(inputTensor)
    val output = model.predict(input)
    val expectedShape = new util.ArrayList[Integer]()
    expectedShape.add(10)
    output.get(0).get(0).getShape should be (expectedShape)
  }

}
