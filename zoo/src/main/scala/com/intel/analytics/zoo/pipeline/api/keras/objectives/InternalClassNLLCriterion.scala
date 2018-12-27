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

// Some variables are accessable to bigdl, has to set the package as bigdl
package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.nn.abstractnn.SizeAverageStatus
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Engine

import scala.concurrent.{Await, Future}
import scala.concurrent.duration.Duration
import scala.reflect.ClassTag

class InternalClassNLLCriterion[@specialized(Float, Double) T: ClassTag]
(weights: Tensor[T] = null, logProbAsInput: Boolean = true,
 paddingValue: Int = -1)(implicit ev: TensorNumeric[T])
  extends ClassNLLCriterion[T](weights, sizeAverage = false, logProbAsInput, paddingValue) {
  @transient
  private var resultsBackward: Array[Future[_]] = null

  private val epsilon: T = ev.fromType(1e-8)

  private val oneMinusEpsilon: T = ev.minus(ev.one, epsilon)
  sizeAverageStatus = SizeAverageStatus.False

  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    require(input.dim() == 1 || input.dim() == 2,
      "ClassNLLCriterion: " +
        ErrorInfo.constrainInputAsVectorOrBatch +
        s"input dim ${input.dim()}")
    gradInput.resizeAs(input)
    gradInput.zero()

    if (input.dim() == 1) {
      require(input.dim() == target.dim(),
        "InternalClassNLLCriterion: " + ErrorInfo.constrainInputDimSameAsTarget +
          s" Input dimension is: ${ input.dim() } , target dimension is: ${ target.dim() }")
      val curTarget = ev.toType[Int](target.valueAt(1))
      if (curTarget == paddingValue) return gradInput
      gradInput.setValue(curTarget, if (weights != null) ev.times(ev.fromType[Int](-1),
        weights.valueAt(curTarget))
      else ev.fromType[Int](-1))
      if (!logProbAsInput) {
        val clipped = ev.clip(input.valueAt(curTarget), epsilon, oneMinusEpsilon)
        gradInput.setValue(curTarget,
          ev.times(gradInput.valueAt(curTarget), ev.inv(clipped)))
      }
    }
    else if (input.dim() == 2) {
      val batchSize = input.size(1)
      val targetSize = target.size()
      target.squeeze()
      if (resultsBackward == null || resultsBackward.length != batchSize) {
        resultsBackward = new Array[Future[_]](batchSize)
      }

      var i = 1
      while (i <= batchSize) {
        val _i = i
        resultsBackward(_i - 1) = Engine.model.invoke(() => {
          val curTarget = ev.toType[Int](target.valueAt(_i))
          if (curTarget != paddingValue) {
            gradInput.setValue(_i, curTarget, if (weights != null) ev.times(ev.fromType[Int](-1),
              weights.valueAt(curTarget))
            else ev.fromType[Int](-1))
            if (!logProbAsInput) {
              val clipped = ev.clip(input.valueAt(_i, curTarget), epsilon, oneMinusEpsilon)
              gradInput.setValue(_i, curTarget,
                ev.times(gradInput.valueAt(_i, curTarget), ev.inv(clipped)))
            }
          }
        })
        i += 1
      }

      i = 0
      while (i < batchSize) {
        Await.result(resultsBackward(i), Duration.Inf)
        i += 1
      }
      target.resize(targetSize)
    }
    gradInput
  }
}

object InternalClassNLLCriterion {
  def apply[@specialized(Float, Double) T: ClassTag](
    weights: Tensor[T] = null,
    logProbAsInput: Boolean = true,
    paddingValue: Int = -1
  )(implicit ev: TensorNumeric[T]) : InternalClassNLLCriterion[T] = {
    new InternalClassNLLCriterion[T](weights, logProbAsInput, paddingValue)
  }
}
