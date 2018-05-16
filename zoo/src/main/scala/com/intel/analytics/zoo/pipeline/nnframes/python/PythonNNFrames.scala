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

package com.intel.analytics.zoo.pipeline.nnframes.python

import java.util.{ArrayList => JArrayList, List => JList}

import com.intel.analytics.bigdl.dataset.{Sample, Transformer}
import com.intel.analytics.bigdl.optim.{OptimMethod, Trigger, ValidationMethod}
import com.intel.analytics.bigdl.python.api.PythonBigDL
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.visualization.{TrainSummary, ValidationSummary}
import com.intel.analytics.bigdl.{Criterion, Module}
import com.intel.analytics.zoo.feature.common._
import com.intel.analytics.zoo.pipeline.nnframes._
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.sql.DataFrame

import scala.collection.JavaConverters._
import scala.reflect.ClassTag

object PythonNNFrames {

  def ofFloat(): PythonNNFrames[Float] = new PythonNNFrames[Float]()

  def ofDouble(): PythonNNFrames[Double] = new PythonNNFrames[Double]()
}

class PythonNNFrames[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonBigDL[T] {

  def nnReadImage(path: String, sc: JavaSparkContext, minParitions: Int): DataFrame = {
    NNImageReader.readImages(path, sc.sc, minParitions)
  }

  def createNNEstimator(
      model: Module[T],
      criterion: Criterion[T],
      sampleTransformer: Preprocessing[(Any, Option[Any]), Sample[T]]
    ): NNEstimator[T] = {
    NNEstimator(model, criterion).setSamplePreprocessing(sampleTransformer)
  }

  def createNNClassifier(
      model: Module[T],
      criterion: Criterion[T],
      samplePreprocessing: Preprocessing[(Any, Option[AnyVal]), Sample[T]]
    ): NNClassifier[T] = {
    NNClassifier(model, criterion).setSamplePreprocessing(samplePreprocessing)
  }

  def createNNModel(
      model: Module[T],
      samplePreprocessing: Preprocessing[Any, Sample[T]]): NNModel[T] = {
    new NNModel(model).setSamplePreprocessing(samplePreprocessing)
  }

  def createNNClassifierModel(
      model: Module[T],
      samplePreprocessing: Preprocessing[Any, Sample[T]]): NNClassifierModel[T] = {
    NNClassifierModel(model).setSamplePreprocessing(samplePreprocessing)
  }

  def setOptimMethod(
      estimator: NNEstimator[T],
      optimMethod: OptimMethod[T]): NNEstimator[T] = {
    estimator.setOptimMethod(optimMethod)
  }

  def setSamplePreprocessing(
      estimator: NNEstimator[T],
      samplePreprocessing: Preprocessing[(Any, Option[AnyVal]), Sample[T]]): NNEstimator[T] = {
    estimator.setSamplePreprocessing(samplePreprocessing)
  }

  def setSamplePreprocessing(
      model: NNModel[T],
      samplePreprocessing: Preprocessing[Any, Sample[T]]): NNModel[T] = {
    model.setSamplePreprocessing(samplePreprocessing)
  }

  def withOriginColumn(imageDF: DataFrame, imageColumn: String, originColumn: String): DataFrame = {
    NNImageSchema.withOriginColumn(imageDF, imageColumn, originColumn)
  }

  def createScalarToTensor(): ScalarToTensor[T] = {
    new ScalarToTensor()
  }

  def createSeqToTensor(size: JArrayList[Int]): SeqToTensor[T] = {
    SeqToTensor(size.asScala.toArray)
  }

  def createArrayToTensor(size: JArrayList[Int]): ArrayToTensor[T] = {
    ArrayToTensor(size.asScala.toArray)
  }

  def createMLlibVectorToTensor(size: JArrayList[Int]): MLlibVectorToTensor[T] = {
    MLlibVectorToTensor(size.asScala.toArray)
  }

  def createImageFeatureToTensor(): ImageFeatureToTensor[T] = {
    ImageFeatureToTensor()
  }

  def createRowToImageFeature(): RowToImageFeature[T] = {
    RowToImageFeature()
  }

  def createFeatureLabelPreprocessing(
      featureTransfomer: Preprocessing[Any, Tensor[T]],
      labelTransformer: Preprocessing[Any, Tensor[T]]
    ): FeatureLabelPreprocessing[Any, Any, Sample[T]] = {
    FeatureLabelPreprocessing(featureTransfomer, labelTransformer)
      .asInstanceOf[FeatureLabelPreprocessing[Any, Any, Sample[T]]]
  }

  def createChainedPreprocessing(list: JList[Preprocessing[Any, Any]]): Preprocessing[Any, Any] = {
    var cur = list.get(0)
    (1 until list.size()).foreach(t => cur = cur -> list.get(t))
    cur
  }

  def createTensorToSample(): TensorToSample[T] = {
    TensorToSample()
  }

  def createToTuple(): ToTuple = {
    ToTuple()
  }

  def createBigDLAdapter(bt: Transformer[Any, Any]): BigDLAdapter[Any, Any] = {
    BigDLAdapter(bt)
  }

  def setTrainSummary(
      estimator: NNEstimator[T],
      summary: TrainSummary
    ): NNEstimator[T] = {
    estimator.setTrainSummary(summary)
  }

  def setValidation(
      estimator: NNEstimator[T],
      trigger: Trigger,
      validationDF: DataFrame,
      vMethods : JList[ValidationMethod[T]],
      batchSize: Int): NNEstimator[T] = {
    estimator.setValidation(trigger, validationDF, vMethods.asScala.toArray, batchSize)
  }

  def setValidationSummary(
      estimator: NNEstimator[T],
      value: ValidationSummary): NNEstimator[T] = {
    estimator.setValidationSummary(value)
  }

  def setNNModelPreprocessing(
      model: NNModel[T],
      sampleTransformer: Preprocessing[Any, Sample[T]]): NNModel[T] = {
    model.setSamplePreprocessing(sampleTransformer)
  }

  def nnEstimatorClearGradientClippingParams(estimator: NNEstimator[T]): Unit = {
    estimator.clearGradientClippingParams()
  }

  def nnEstimatorSetConstantGradientClipping(
    estimator: NNEstimator[T],
    min: Float,
    max: Float): Unit = {
    estimator.setConstantGradientClipping(min, max)
  }

  def nnEstimatorSetGradientClippingByL2Norm(
    estimator: NNEstimator[T],
    clipNorm: Float): Unit = {
    estimator.setGradientClippingByL2Norm(clipNorm)
  }
}
