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

import java.util.{List => JList, Map => JMap}

import com.intel.analytics.bigdl.dataset.PaddingParam
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.python.api.{PythonBigDL, Sample}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image.{BytesToMat, FeatureTransformer, MatToFloats}
import com.intel.analytics.zoo.feature.image.ImageSet
import com.intel.analytics.zoo.models.common.ZooModel
import com.intel.analytics.zoo.models.image.common.{ImageConfigure, ImageModel}
import com.intel.analytics.zoo.models.objectdetection._
import com.intel.analytics.zoo.models.recommendation.{NeuralCF, Recommender, UserItemFeature, UserItemPrediction}
import com.intel.analytics.zoo.models.textclassification.TextClassifier
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag
import scala.collection.JavaConverters._

object PythonZooModel {

  def ofFloat(): PythonZooModel[Float] = new PythonZooModel[Float]()

  def ofDouble(): PythonZooModel[Double] = new PythonZooModel[Double]()
}

class PythonZooModel[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonBigDL[T] {

  def saveZooModel(
      model: ZooModel[Activity, Activity, T],
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

  def loadObjectDetector(path: String, weightPath: String = null): ObjectDetector[T] = {
    ObjectDetector.loadModel(path, weightPath)
  }

  def loadImageModel(path: String, weightPath: String = null): ImageModel[T] = {
    ImageModel.loadModel(path, weightPath)
  }

  def readPascalLabelMap(): JMap[Int, String] = {
    LabelReader.readPascalLabelMap().asJava
  }

  def readCocoLabelMap(): JMap[Int, String] = {
    LabelReader.readCocoLabelMap().asJava
  }

  def imageModelPredict(model: ImageModel[T],
    image: ImageSet,
    config: ImageConfigure[T] = null): ImageSet = {
    model.predictImageSet(image, config)
  }

  def getImageConfig(model: ImageModel[T]): ImageConfigure[T] = {
    model.getConfig
  }

  def createImageConfigure(preProcessor: FeatureTransformer,
                           postProcessor: FeatureTransformer,
                           batchPerPartition: Int,
                           labelMap: JMap[Int, String],
                           paddingParam: PaddingParam[T]): ImageConfigure[T] = {
    val map = if (labelMap == null) null else labelMap.asScala.toMap
    ImageConfigure(preProcessor, postProcessor, batchPerPartition, map, Option(paddingParam))
  }

  def createVisualizer(labelMap: JMap[Int, String], thresh: Float = 0.3f,
                       encoding: String): FeatureTransformer = {
    Visualizer(labelMap.asScala.toMap, thresh, encoding, Visualizer.visualized) ->
      BytesToMat(Visualizer.visualized) -> MatToFloats(shareBuffer = false)
  }

  def getLabelMap(imageConfigure: ImageConfigure[T]): JMap[Int, String] = {
    if (imageConfigure.labelMap == null) null else imageConfigure.labelMap.asJava
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

  def createPaddingParam(): PaddingParam[T] = {
    PaddingParam()
  }

  def createZooNeuralCF(
      userCount: Int,
      itemCount: Int,
      numClasses: Int,
      userEmbed: Int = 20,
      itemEmbed: Int = 20,
      hiddenLayers: JList[Int],
      includeMF: Boolean = true,
      mfEmbed: Int = 20): NeuralCF[T] = {
    NeuralCF[T](userCount, itemCount, numClasses, userEmbed, itemEmbed,
      hiddenLayers.asScala.toArray, includeMF, mfEmbed)
  }

  def loadNeuralCF(
      path: String,
      weightPath: String = null): NeuralCF[T] = {
    NeuralCF.loadModel(path, weightPath)
  }

  def toUserItemFeatureRdd(featureRdd: JavaRDD[Array[Object]]): RDD[UserItemFeature[T]] = {
    featureRdd.rdd.foreach(x =>
      require(x.length == 3, "UserItemFeature should consist of userId, itemId and sample"))
    featureRdd.rdd.map(x =>
      UserItemFeature(x(0).asInstanceOf[Int], x(1).asInstanceOf[Int],
        toJSample(x(2).asInstanceOf[Sample])))
  }

  def toPredictionJavaRdd(predictionRdd: RDD[UserItemPrediction]): JavaRDD[JList[Double]] = {
    predictionRdd.map(x =>
      List(x.userId.toDouble, x.itemId.toDouble, x.prediction.toDouble, x.probability)
        .asJava).toJavaRDD()
  }

  def predictUserItemPair(
      model: Recommender[T],
      featureRdd: JavaRDD[Array[Object]]): JavaRDD[JList[Double]] = {
    val predictionRdd = model.predictUserItemPair(toUserItemFeatureRdd(featureRdd))
    toPredictionJavaRdd(predictionRdd)
  }

  def recommendForUser(
      model: Recommender[T],
      featureRdd: JavaRDD[Array[Object]],
      maxItems: Int): JavaRDD[JList[Double]] = {
    val predictionRdd = model.recommendForUser(toUserItemFeatureRdd(featureRdd), maxItems)
    toPredictionJavaRdd(predictionRdd)
  }

  def recommendForItem(
      model: Recommender[T],
      featureRdd: JavaRDD[Array[Object]],
      maxUsers: Int): JavaRDD[JList[Double]] = {
    val predictionRdd = model.recommendForItem(toUserItemFeatureRdd(featureRdd), maxUsers)
    toPredictionJavaRdd(predictionRdd)
  }

}
