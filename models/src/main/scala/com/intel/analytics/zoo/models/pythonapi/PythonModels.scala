/*
 * Copyright 2016 The BigDL Authors.
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

package com.intel.analytics.zoo.models.pythonapi

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.python.api.PythonBigDL
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image._
import com.intel.analytics.zoo.models.Configure
import java.util.{Map => JMap}

import com.intel.analytics.bigdl.dataset.PaddingParam
import com.intel.analytics.bigdl.zoo.models.Predictor
import com.intel.analytics.zoo.models.imageclassification.util.LabelOutput
import com.intel.analytics.zoo.models.objectdetection.utils._

import scala.collection.JavaConverters._
import org.apache.log4j.Logger

import scala.reflect.ClassTag

object PythonModels {

  def ofFloat(): PythonBigDL[Float] = new PythonModels[Float]()

  def ofDouble(): PythonBigDL[Double] = new PythonModels[Double]()

  val logger = Logger.getLogger(getClass)
}

class PythonModels[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonBigDL[T] {

   def modelZooPredict(predictor: Predictor[T],
     imageFrame: ImageFrame,
     outputLayer: String = null,
     shareBuffer: Boolean = false,
     predictKey: String = ImageFeature.predict): ImageFrame = {
     predictor.predict(imageFrame, outputLayer, shareBuffer, predictKey)
   }

  def createPredictor(model: AbstractModule[Activity, Activity, T],
    configure: Configure[T]): Predictor[T] = {
    Predictor(model, configure)
  }

  def createConfigure(preProcessor: FeatureTransformer,
    postProcessor: FeatureTransformer,
    batchPerPartition: Int,
    labelMap: JMap[Int, String],
    paddingParam: PaddingParam[T]): Configure[T] = {
    val map = if (labelMap == null) null else labelMap.asScala.toMap
    Configure(preProcessor, postProcessor, batchPerPartition, map, Option(paddingParam))
  }

  def createVisualizer(labelMap: JMap[Int, String], thresh: Float = 0.3f,
    encoding: String): FeatureTransformer = {
    Visualizer(labelMap.asScala.toMap, thresh, encoding, Visualizer.visualized) ->
    BytesToMat(Visualizer.visualized) -> MatToFloats(shareBuffer = false)
  }

  def getConfigure(predictor: Predictor[T]): Configure[T] = {
    predictor.configure
  }

  def getLabelMap(configure: Configure[T]): JMap[Int, String] = {
    if (configure.labelMap == null) null else configure.labelMap.asJava
  }

  def readPascalLabelMap(): JMap[Int, String] = {
    LabelReader.readPascalLabelMap().asJava
  }

  def readCocoLabelMap(): JMap[Int, String] = {
    LabelReader.readCocoLabelMap().asJava
  }

  def createImInfo(): ImInfo = {
    ImInfo()
  }

  def createDecodeOutput(): DecodeOutput = {
    DecodeOutput()
  }

  def createScaleDetection(): ScaleDetection = {
    ScaleDetection()
  }

  def createLabelOutput(labelMap: JMap[Int, String], clses: String,
                       probs: String): FeatureTransformer = {
    LabelOutput(labelMap.asScala.toMap, clses, probs)
  }

  def createPaddingParam(): PaddingParam[T] = {
    PaddingParam()
  }
}
