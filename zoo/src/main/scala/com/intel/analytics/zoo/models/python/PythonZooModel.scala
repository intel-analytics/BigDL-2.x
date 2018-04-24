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

package com.intel.analytics.zoo.models.python

import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.python.api.PythonBigDL
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.models.common.ZooModel
import com.intel.analytics.zoo.models.textclassification.TextClassifier

import scala.reflect.ClassTag

object PythonZooModel {

  def ofFloat(): PythonZooModel[Float] = new PythonZooModel[Float]()

  def ofDouble(): PythonZooModel[Double] = new PythonZooModel[Double]()
}

class PythonZooModel[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonBigDL[T] {

  def saveZooModel(model: ZooModel[Activity, Activity, T],
                path: String,
                weightPath: String = null,
                overWrite: Boolean = false): ZooModel[Activity, Activity, T] = {
    model.saveModel(path, weightPath, overWrite)
  }

  def createZooTextClassifier(
      classNum: Int,
      tokenLength: Int,
      sequenceLength: Int = 500,
      encoder: String = "cnn",
      encoderOutputDim: Int = 256): TextClassifier[T] = {
    TextClassifier[T](classNum, tokenLength, sequenceLength, encoder, encoderOutputDim)
  }

  def loadTextClassifier(
      path: String,
      weightPath: String = null): TextClassifier[T] = {
    TextClassifier.loadModel(path, weightPath)
  }

}
