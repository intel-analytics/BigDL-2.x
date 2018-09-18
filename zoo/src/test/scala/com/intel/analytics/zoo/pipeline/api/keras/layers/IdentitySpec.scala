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

package com.intel.analytics.zoo.pipeline.api.keras.layers

import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.zoo.pipeline.api.keras.layers.{Identity => ZIdentity}
import com.intel.analytics.bigdl.nn.{Module, Identity => BIdentity}
import scala.util.Random
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{RandomGenerator, Shape, T, Table}
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest
import org.scalatest.{FlatSpec, Matchers}

class IdentitySpec extends KerasBaseSpec {
  "Identity Zoo" should "be the same as BigDL" in {
    val blayer = BIdentity[Float]()
    val zlayer = ZIdentity[Float](inputShape = Shape(Array(3)))
    zlayer.build(Shape(Array(-1, 3)))
    assert(zlayer.getOutputShape() == Shape(-1, 3))
    val input = Tensor[Float](Array(2, 3)).rand()
    compareOutputAndGradInput(
      blayer.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      zlayer.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      input)
  }
}

class IdentitySerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = ZIdentity[Float](inputShape = Shape(Array(3)))
    layer.build(Shape(List(Shape(-1, 3))))
    val input = Tensor[Float](Array(2, 3)).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }
}
