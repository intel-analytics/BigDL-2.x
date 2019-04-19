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

package com.intel.analytics.zoo.models.common

import com.intel.analytics.bigdl.{Criterion, DataSet}
import com.intel.analytics.bigdl.dataset.{LocalDataSet, MiniBatch, PaddingParam, Sample}
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.optim.{OptimMethod, ValidationMethod, ValidationResult}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.feature.image.ImageSet
import com.intel.analytics.zoo.feature.text.TextSet
import com.intel.analytics.zoo.pipeline.api.autograd.Variable
import com.intel.analytics.zoo.pipeline.api.keras.models.KerasNet
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

abstract class KerasZooModel[A <: Activity : ClassTag, B <: Activity : ClassTag, T: ClassTag]
(implicit ev: TensorNumeric[T]) extends ZooModel[A, B, T] {

  // This class is defined to leverage APIs of KerasNet
  // For the following methods, please refer to KerasNet for documentation.

  def compile(
      optimizer: OptimMethod[T],
      loss: Criterion[T],
      metrics: List[ValidationMethod[T]] = null)(implicit ev: TensorNumeric[T]): Unit = {
    model.asInstanceOf[KerasNet[T]].compile(optimizer, loss, metrics)
  }

  def compile(
      optimizer: String,
      loss: String,
      metrics: List[String])(implicit ev: TensorNumeric[T]): Unit = {
    model.asInstanceOf[KerasNet[T]].compile(optimizer, loss, metrics)
  }

  def compile(
      optimizer: String,
      loss: String)(implicit ev: TensorNumeric[T]): Unit = {
    model.asInstanceOf[KerasNet[T]].compile(optimizer, loss)
  }

  def compile(
      optimizer: OptimMethod[T],
      loss: (Variable[T], Variable[T]) => Variable[T],
      metrics: List[ValidationMethod[T]])(implicit ev: TensorNumeric[T]): Unit = {
    model.asInstanceOf[KerasNet[T]].compile(optimizer, loss, metrics)
  }

  def compile(
      optimizer: OptimMethod[T],
      loss: (Variable[T], Variable[T]) => Variable[T])(implicit ev: TensorNumeric[T]): Unit = {
    model.asInstanceOf[KerasNet[T]].compile(optimizer, loss)
  }

  def setTensorBoard(logDir: String, appName: String): Unit = {
    model.asInstanceOf[KerasNet[T]].setTensorBoard(logDir, appName)
  }

  def getTrainSummary(tag: String): Array[(Long, Float, Double)] = {
    model.asInstanceOf[KerasNet[T]].getTrainSummary(tag: String)
  }

  def getValidationSummary(tag: String): Array[(Long, Float, Double)] = {
    model.asInstanceOf[KerasNet[T]].getValidationSummary(tag: String)
  }

  def setCheckpoint(path: String, overWrite: Boolean = true): Unit = {
    model.asInstanceOf[KerasNet[T]].setCheckpoint(path, overWrite)
  }

  def clearGradientClipping(): Unit = {
    model.asInstanceOf[KerasNet[T]].clearGradientClipping()
  }

  def setConstantGradientClipping(min: Float, max: Float): Unit = {
    model.asInstanceOf[KerasNet[T]].setConstantGradientClipping(min, max)
  }

  def setGradientClippingByL2Norm(clipNorm: Float): Unit = {
    model.asInstanceOf[KerasNet[T]].setGradientClippingByL2Norm(clipNorm)
  }

  def fit(
      x: DataSet[MiniBatch[T]],
      nbEpoch: Int,
      validationData: DataSet[MiniBatch[T]])(implicit ev: TensorNumeric[T]): Unit = {
    model.asInstanceOf[KerasNet[T]].fit(x, nbEpoch, validationData)
  }

  def fit(
      x: DataSet[MiniBatch[T]],
      nbEpoch: Int)(implicit ev: TensorNumeric[T]): Unit = {
    model.asInstanceOf[KerasNet[T]].fit(x, nbEpoch)
  }

  def fit(
      x: RDD[Sample[T]],
      batchSize: Int = 32,
      nbEpoch: Int = 10,
      validationData: RDD[Sample[T]] = null,
      featurePaddingParam: PaddingParam[T] = null,
      labelPaddingParam: PaddingParam[T] = null)(implicit ev: TensorNumeric[T]): Unit = {
    model.asInstanceOf[KerasNet[T]]
      .fit(x, batchSize, nbEpoch, validationData, featurePaddingParam, labelPaddingParam)
  }

  def fit(
      x: ImageSet,
      batchSize: Int,
      nbEpoch: Int,
      validationData: ImageSet)(implicit ev: TensorNumeric[T]): Unit = {
    model.asInstanceOf[KerasNet[T]].fit(x, batchSize, nbEpoch, validationData)
  }

  def fit(
      x: ImageSet,
      batchSize: Int,
      nbEpoch: Int)(implicit ev: TensorNumeric[T]): Unit = {
    model.asInstanceOf[KerasNet[T]].fit(x, batchSize, nbEpoch)
  }

  def fit(
      x: TextSet,
      batchSize: Int,
      nbEpoch: Int,
      validationData: TextSet)(implicit ev: TensorNumeric[T]): Unit = {
    model.asInstanceOf[KerasNet[T]].fit(x, batchSize, nbEpoch, validationData)
  }

  def fit(
      x: TextSet,
      batchSize: Int,
      nbEpoch: Int)(implicit ev: TensorNumeric[T]): Unit = {
    model.asInstanceOf[KerasNet[T]].fit(x, batchSize, nbEpoch)
  }

  def evaluate(
      x: RDD[Sample[T]],
      batchSize: Int)
      (implicit ev: TensorNumeric[T]): Array[(ValidationResult, ValidationMethod[T])] = {
    model.asInstanceOf[KerasNet[T]].evaluate(x, batchSize)
  }

  def evaluate(x: LocalDataSet[MiniBatch[T]])
      (implicit ev: TensorNumeric[T]): Array[(ValidationResult, ValidationMethod[T])] = {
    model.asInstanceOf[KerasNet[T]].evaluate(x)
  }

  def evaluate(
      x: ImageSet,
      batchSize: Int)
      (implicit ev: TensorNumeric[T]): Array[(ValidationResult, ValidationMethod[T])] = {
    model.asInstanceOf[KerasNet[T]].evaluate(x, batchSize)
  }

  def evaluate(
      x: TextSet,
      batchSize: Int): Array[(ValidationResult, ValidationMethod[T])] = {
    model.asInstanceOf[KerasNet[T]].evaluate(x, batchSize)
  }

  def predict(
      x: RDD[Sample[T]],
      batchPerThread: Int): RDD[Activity] = {
    model.asInstanceOf[KerasNet[T]].predict(x, batchPerThread)
  }
}
