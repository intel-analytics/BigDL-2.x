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

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.AbstractCriterion
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Shape, T}
import com.intel.analytics.zoo.pipeline.api.autograd.{AutoGrad => A}
import com.intel.analytics.zoo.pipeline.api.keras.layers.KerasBaseSpec
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.KerasUtils

import scala.reflect.ClassTag

// scalastyle:off
class CustomizedLossSpec extends KerasBaseSpec {

  // inputShape shape including batch
  private def compareLossesWithTorch(a: AbstractCriterion[Tensor[Float], Tensor[Float], Float],
      b: AbstractCriterion[Tensor[Float], Tensor[Float], Float],
      inputShape: Shape): Unit = {
    val input = Tensor[Float](inputShape.toSingle().toArray).rand(0, 1).fill(1f)
    val target = Tensor[Float](inputShape.toSingle().toArray).rand(0, 1).fill(1f)
    val aOut = a.forward(input, target)
    val bOut = b.forward(input, target)
    assert(aOut === bOut)

    val aGrad = a.backward(input, target)
    val bGrad = b.backward(input, target)
    assert(aGrad.almostEqual(bGrad, 1e-5), s"aGrad: $aGrad, bGrad: $bGrad")
  }

  "self defined abs loss" should "be test" in {

    def cLoss[T: ClassTag](yTrue: Variable[T], yPred: Variable[T])(
        implicit ev: TensorNumeric[T]): Variable[T] = {
      A.mean(A.abs(yTrue - yPred), axis = 1)
    }
    val targetShape = Shape(2, 3)
    val loss = CustomLoss[Float](cLoss[Float], KerasUtils.removeBatch(targetShape))
    compareLossesWithTorch(loss, AbsCriterion[Float](), targetShape)
  }

  "self defined poisson loss" should "be test" in {

    def poisson[T: ClassTag](yTrue: Variable[T], yPred: Variable[T])(
        implicit ev: TensorNumeric[T]): Variable[T] = {
      A.mean(yPred - yTrue * A.log(yPred + A.epsilon()), axis = 1)
    }
    val targetShape = Shape(6, 7)
    val loss = CustomLoss[Float](poisson[Float], yPredShape = KerasUtils.removeBatch(targetShape),
      yTrueShape = KerasUtils.removeBatch(targetShape))
      compareLossesWithTorch(loss, PoissonCriterion[Float](), targetShape)
  }

  "self defined categorical crossentropy" should "be ok" in {
    ifskipTest()
    val kerasCode =
      """
        |input_tensor = Input(shape=[3])
        |target_tensor = Input(shape=[3])
        |loss = categorical_crossentropy(target_tensor, input_tensor)
        |input = np.random.uniform(0, 1, [2, 3])
        |Y = np.zeros((2, 3))
        |index = np.array([1, 2])
        |Y[np.arange(2), index] = 1
      """.stripMargin

    def lossFunc[T: ClassTag](yTrue: Variable[T], yPred: Variable[T])(
        implicit ev: TensorNumeric[T]): Variable[T] = {
      var output = yPred / (A.sum(yPred, axis = 1, keepDims = true)) // .broadcast(axis = 1, times = 3)
      output = A.clip(output, AutoGrad.epsilon(), 1.0 - AutoGrad.epsilon())
      -A.sum(A.log(output) * yTrue, axis = 1)
    }
    val loss = CustomLoss[Float](lossFunc[Float], Shape(3))
    checkOutputAndGradForLoss(loss, kerasCode)
  }

  "mean_squared_logarithmic_error" should "be ok" in {
    ifskipTest()
    val kerasCode =
      """
        |input_tensor = Input(shape=[3])
        |target_tensor = Input(shape=[3])
        |loss = mean_squared_logarithmic_error(target_tensor, input_tensor)
        |input = np.random.uniform(0, 1, [2, 3])
        |Y = np.random.uniform(0, 1, [2, 3])
      """.stripMargin

    def lossFunc[T: ClassTag](yTrue: Variable[T], yPred: Variable[T])(
        implicit ev: TensorNumeric[T]): Variable[T] = {
      val first_log = A.log(A.clip(yPred, A.epsilon(), Double.MaxValue) + 1.0)
      val second_log = A.log(A.clip(yTrue, A.epsilon(), Double.MaxValue) + 1.0)
      return A.mean(A.square(first_log - second_log), axis = 1)
    }
    val loss = CustomLoss[Float](lossFunc[Float], Shape(3))
    checkOutputAndGradForLoss(loss, kerasCode)
  }

  "squared_hinge" should "be ok" in {
    ifskipTest()
    val kerasCode =
      """
        |input_tensor = Input(shape=[3])
        |target_tensor = Input(shape=[3])
        |loss = squared_hinge(target_tensor, input_tensor)
        |input = np.random.uniform(0, 1, [2, 3])
        |Y = np.random.uniform(0, 1, [2, 3])
      """.stripMargin

    def lossFunc[T: ClassTag](yTrue: Variable[T], yPred: Variable[T])(
        implicit ev: TensorNumeric[T]): Variable[T] = {
      A.mean(A.square(A.maximum(-yTrue * yPred + 1, 0)), axis = 1)
    }
    val loss = CustomLoss[Float](lossFunc[Float], Shape(3))
    checkOutputAndGradForLoss(loss, kerasCode)
  }

  "python api " should "be test" in {
    val yTrue = Variable[Float](inputShape = Shape(3))
    val yPred = Variable[Float](inputShape = Shape(3))
    val input = Tensor[Float](2, 3).rand(0, 1)
    val target = Tensor[Float](2, 3).rand(0, 1)
    new CustomLossWithVariable[Float](Array(yTrue, yPred),
      A.mean(A.abs(yTrue - yPred), axis = 1)).forward(input, target)
  }

  "broadcast (2, 3, 4) with (2, 3, 4)" should "be test" in {
    val x = Variable[Float](inputShape = Shape(3, 4))
    val y = Variable[Float](inputShape = Shape(3, 4))
    val r = x + y
    val xValue = x.getDummyTensor(1, batchSize = 2)
    val yValue = y.getDummyTensor(1, batchSize = 2)
    val zr = r.toGraph(Array(x, y)).forward(T(xValue, yValue))
    assert(zr.toTensor[Float].almostEqual(x.getDummyTensor(2, batchSize = 2), 1e-6))
  }

  "broadcast (2, 3, 4) with (2, 3, 1)" should "be test" in {
    val x = Variable[Float](inputShape = Shape(3, 4))
    val y = Variable[Float](inputShape = Shape(3, 1))
    val r = x + y
    val xValue = x.getDummyTensor(1, batchSize = 2)
    val yValue = y.getDummyTensor(1, batchSize = 2)
    val zr = r.toGraph(Array(x, y)).forward(T(xValue, yValue))
    assert(zr.toTensor[Float].almostEqual(x.getDummyTensor(2, batchSize = 2), 1e-6))
  }

  "broadcast (2, 3, 1) with (2, 3, 4)" should "be test" in {
    val x = Variable[Float](inputShape = Shape(3, 1))
    val y = Variable[Float](inputShape = Shape(3, 4))
    val r = x + y
    val xValue = x.getDummyTensor(1, batchSize = 2)
    val yValue = y.getDummyTensor(1, batchSize = 2)
    val zr = r.toGraph(Array(x, y)).forward(T(xValue, yValue))
    assert(zr.toTensor[Float].almostEqual(y.getDummyTensor(2, batchSize = 2), 1e-6))
  }

  "broadcast (2, 1, 4) with (2, 3, 4)" should "be test" in {
    val x = Variable[Float](inputShape = Shape(1, 4))
    val y = Variable[Float](inputShape = Shape(3, 4))
    val r = x + y
    val xValue = x.getDummyTensor(1, batchSize = 2)
    val yValue = y.getDummyTensor(1, batchSize = 2)
    val zr = r.toGraph(Array(x, y)).forward(T(xValue, yValue))
    assert(zr.toTensor[Float].almostEqual(y.getDummyTensor(2, batchSize = 2), 1e-6))
  }

  "broadcast (2, 2, 4) with (2, 3, 4)" should "be test" in {
    intercept[Exception] {
      val x = Variable[Float](inputShape = Shape(2, 4))
      val y = Variable[Float](inputShape = Shape(3, 4))
      val r = x + y
      val xValue = Tensor[Float](x.getInputShape().copyAndUpdate(0, 2).toSingle().toArray).rand()
      val yValue = Tensor[Float](y.getInputShape().copyAndUpdate(0, 2).toSingle().toArray).rand()
      r.toGraph(Array(x, y)).forward(T(xValue, yValue))
    }
  }
}
