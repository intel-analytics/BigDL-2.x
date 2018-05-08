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

package com.intel.analytics.zoo.pipeline.api.autograd

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Shape, T, Table}
import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.api.keras.layers.{Input, InputLayer, KerasBaseSpec}
import com.intel.analytics.zoo.pipeline.api.keras.models.{Model, Sequential}

import scala.reflect.ClassTag

class LambdaSpec extends KerasBaseSpec{

  "1 input layer with inputShape" should "be test" in {
    def lambdaFunc[T: ClassTag](a: List[Variable[T]])(
        implicit ev: TensorNumeric[T]): Variable[T] = {
      a(0) + 2
    }
    val seq = Sequential[Float]()
    val layer = Lambda[Float](lambdaFunc[Float], inputShape = Shape(3))
    seq.add(layer)

    val kerasCode =
      """
        |input_tensor = Input(shape=[3])
        |input = np.random.uniform(0, 1, [1, 3])
        |output_tensor = Lambda(lambda x: x + 2)(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin

    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode)
  }

  "1 input layer without inputShape" should "be test" in {
    def lambdaFunc[T: ClassTag](a: List[Variable[T]])(
        implicit ev: TensorNumeric[T]): Variable[T] = {
      a(0) + 2
    }
    val seq = Sequential[Float]()
    seq.add(InputLayer[Float](Shape(3)))
    val layer = Lambda[Float](lambdaFunc[Float])
    seq.add(layer)

    val kerasCode =
      """
        |input_tensor = Input(shape=[3])
        |input = np.random.uniform(0, 1, [1, 3])
        |output_tensor = Lambda(lambda x: x + 2)(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin

    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode)
  }


  "two inputs layer" should "be test" in {
    def lambdaFunc[T: ClassTag](inputs: List[Variable[T]])(
        implicit ev: TensorNumeric[T]): Variable[T] = {
      inputs(0) + inputs(1)
    }

    val input1 = Input[Float](Shape(3))
    val input2 = Input[Float](Shape(3))
    val layer = Lambda[Float](lambdaFunc[Float]).inputs(input1, input2)
    val model = Model[Float](Array(input1, input2), layer)
    val tmpFile = createTmpFile()

    model.saveModule(tmpFile.getAbsolutePath, overWrite = true)
    val reloadModel = Net.load[Float](tmpFile.getAbsolutePath)
    compareOutputAndGradInputTable2Tensor(
      model1 = model.asInstanceOf[AbstractModule[Table, Tensor[Float], Float]],
      model2 = reloadModel.asInstanceOf[AbstractModule[Table, Tensor[Float], Float]],
      input = T(Tensor[Float](2, 3).rand(), Tensor[Float](2, 3).rand()))
  }
}
