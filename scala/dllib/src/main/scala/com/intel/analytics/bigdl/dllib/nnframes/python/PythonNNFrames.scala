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

import java.util.{ArrayList => JArrayList}

import com.intel.analytics.bigdl.{Criterion, Module}
import com.intel.analytics.bigdl.python.api.PythonBigDL
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image.FeatureTransformer
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

  def createNNImageTransformer(transformer: FeatureTransformer): NNImageTransformer = {
    new NNImageTransformer(transformer)
  }

  def createNNEstimator(
        model: Module[T],
        criterion: Criterion[T],
        featureSize: JArrayList[Int],
        labelSize: JArrayList[Int]): NNEstimator[T] = {
    new NNEstimator[T](model, criterion, featureSize.asScala.toArray, labelSize.asScala.toArray)
  }

  def createNNClassifier(
        model: Module[T],
        criterion: Criterion[T],
        featureSize: JArrayList[Int],
        labelSize: JArrayList[Int]): NNClassifier[T] = {
    new NNClassifier[T](model, criterion, featureSize.asScala.toArray)
  }

  def createNNModel(model: Module[T], featureSize: JArrayList[Int]): NNModel[T] = {
    new NNModel[T](model, featureSize.asScala.toArray)
  }

  def createNNClassifierModel(
      model: Module[T],
      featureSize: JArrayList[Int]): NNClassifierModel[T] = {
    new NNClassifierModel[T](model, featureSize.asScala.toArray)
  }

  def setFeatureSizeNNModel(nnModel: NNModel[T], featureSize: JArrayList[Int]): NNModel[T] = {
    nnModel.setFeatureSize(featureSize.asScala.toArray)
  }

  def setFeatureSizeNNClassifierModel(
        nnClassifierModel: NNClassifierModel[T],
        featureSize: JArrayList[Int]): NNClassifierModel[T] = {
    nnClassifierModel.setFeatureSize(featureSize.asScala.toArray)
  }
}
