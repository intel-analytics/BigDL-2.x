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

package com.intel.analytics.zoo.pipeline.api.keras2.layers

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest

import scala.util.Random

class GlobalAveragePooling1D extends KerasBaseSpec{

  """Global average pooling operation for temporal data.

    # Input shape
        3D tensor with shape: `(samples, steps, features)`.

    # Output shape
        2D tensor with shape: `(samples, features)`.
    """

  "GlobalAveragePooling1D" should "be the same as Keras" in {
    val kerasCode =
      """
        |inputTensor = Input(shape=[3, 24])
        |input = np.random.random([2, 3, 24])
        |outputTensor = GlobalAveragePooling1D()(inputTensor)
        |model = Model(input=inputTensor, output=outputTensor)
      """.stripMargin
    val seq = Sequential[Float]()
    val layer = GlobalAveragePooling1D[Float](inputShape = Shape(3, 24))
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 24))
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode)
  }

}

class GlobalAveragePooling1DSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = GlobalAveragePooling1D[Float](inputShape = Shape(3, 24))
    layer.build(Shape(2, 3, 24))
    val input = Tensor[Float](2, 3, 24).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }
}
