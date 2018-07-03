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

package com.intel.analytics.zoo.pipeline.api.keras.objectives

import com.intel.analytics.bigdl.nn.abstractnn.TensorCriterion
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion => BigDLClassNLLCriterion}

import scala.reflect.ClassTag

/**
 * The negative log likelihood criterion.
 * It is useful to train a classification problem with n classes.
 * If provided, the optional argument weights should be a 1D Tensor assigning weight to
 * each of the classes. This is particularly useful when you have an unbalanced training set.
 *
 * The input given through a forward() is expected to contain log-probabilities/probabilities of
 * each class: input has to be a 1D Tensor of size n.
 * Obtaining log-probabilities/probabilities in a neural network is easily achieved by
 * adding a LogSoftMax/SoftMax layer in the last layer of your neural network.
 * You may use CrossEntropyCriterion instead, if you prefer not to add an extra layer
 * to your network.
 *
 * In the log-probabilities case,
 * The loss can be described as:
 *     loss(x, class) = -x[class]
 * or in the case of the weights argument it is specified as follows:
 *     loss(x, class) = -weights[class] * x[class]
 *
 * Due to the behaviour of the backend code, it is necessary to set sizeAverage to false when
 * calculating losses in non-batch mode.
 *
 * Note that if the target is `paddingValue`, the training process will skip this sample.
 * In other words, the forward process will return zero output and the backward process
 * will also return zero `gradInput`.
 *
 * By default, the losses are averaged over observations for each minibatch.
 * However, if the field sizeAverage is set to false, the losses are instead
 * summed for each minibatch.
 *
 * In particular, when weights=null, size_average=true and logProbAsInput=false, this is same as
 * `sparse_categorical_crossentropy` loss in keras.
 *
 * @param weights weights of each element of the input
 * @param sizeAverage size average of batch
 * @param logProbAsInput indicating whether to accept log-probabilities or probabilities as input.
 *                   True means accepting log-probabilities as input.
 * @param ev numeric operator
 * @tparam T numeric type
 */
class ClassNLLCriterion[T: ClassTag](
    val weights: Tensor[T] = null,
    val sizeAverage: Boolean = true,
    val logProbAsInput: Boolean = true,
    val paddingValue: Int = -1,
    val zeroBasedLabel: Boolean = true)(implicit ev: TensorNumeric[T])
  extends NetTensorCriterion[T] {

  override val loss: TensorCriterion[T] =
    BigDLClassNLLCriterion[T](weights, sizeAverage, logProbAsInput, paddingValue)

  private val buffer: Tensor[T] = Tensor[T]()

  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    if (zeroBasedLabel) {
      buffer.resizeAs(target)
      buffer.fill(ev.one).add(target)
      output = loss.updateOutput(input, buffer)
      output
    }
    else {
      output = loss.updateOutput(input, target)
      output
    }
  }

  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    if (zeroBasedLabel) {
      buffer.resizeAs(target)
      buffer.fill(ev.one).add(target)
      gradInput = loss.updateGradInput(input, buffer)
      gradInput
    }
    else {
      gradInput = loss.updateGradInput(input, target)
      gradInput
    }
  }
}

object ClassNLLCriterion {
  def apply[@specialized(Float, Double) T: ClassTag](
      weights: Tensor[T] = null,
      sizeAverage: Boolean = true,
      logProbAsInput: Boolean = true,
      paddingValue: Int = -1,
      zeroBasedLabel: Boolean = true)
    (implicit ev: TensorNumeric[T]): ClassNLLCriterion[T] = {
    new ClassNLLCriterion[T](weights, sizeAverage, logProbAsInput,
      paddingValue, zeroBasedLabel)
  }
}
