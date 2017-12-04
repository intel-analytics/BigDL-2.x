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

import java.util.{ArrayList => JArrayList, HashMap => JHashMap, List => JList, Map => JMap}

import com.intel.analytics.bigdl.python.api.PythonBigDL
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.models.Predictor
import com.intel.analytics.zoo.models.alexnet.AlexnetPredictor
import com.intel.analytics.zoo.models.dataset.PredictResult
import com.intel.analytics.zoo.models.inception.{DensenetPredictor, InceptionV1Predictor}
import com.intel.analytics.zoo.models.mobilenet.MobilenetPredictor
import com.intel.analytics.zoo.models.resnet.ResnetPredictor
import com.intel.analytics.zoo.models.squeezenet.SqueezenetPredictor
import com.intel.analytics.zoo.models.vgg.VGGPredictor
import com.intel.analytics.zoo.transform.vision.pythonapi.PythonVisionTransform
import org.apache.log4j.Logger
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SQLContext}

import scala.reflect.ClassTag
import scala.collection.JavaConverters._

object ImageNetPredictor {

  def ofFloat(): PythonBigDL[Float] = new ImageNetPredictor[Float]()

  def ofDouble(): PythonBigDL[Double] = new ImageNetPredictor[Double]()

  val logger = Logger.getLogger(getClass)
}

class ImageNetPredictor[T: ClassTag](implicit ev: TensorNumeric[T])
  extends PythonVisionTransform[T] {

  def createAlexnetPredictor(modelPath: String, meanPath : String): AlexnetPredictor = {
    AlexnetPredictor(modelPath, meanPath)
  }

  def createInceptionV1Predictor(modelPath: String): InceptionV1Predictor = {
    InceptionV1Predictor(modelPath)
  }

  def createResnetPredictor(modelPath: String): ResnetPredictor = {
    ResnetPredictor(modelPath)
  }

  def createVGGPredictor(modelPath: String): VGGPredictor = {
    VGGPredictor(modelPath)
  }

  def createDensenetPredictor(modelPath: String): DensenetPredictor = {
    DensenetPredictor(modelPath)
  }

  def createMobilenetPredictor(modelPath: String): MobilenetPredictor = {
    MobilenetPredictor(modelPath)
  }

  def createSqueezenetPredictor(modelPath: String): SqueezenetPredictor = {
    SqueezenetPredictor(modelPath)
  }

  def predictLocal(predictor: Predictor, imagePath : String, topN : Int) :
    Array[JList[_]] = {
    val predictResult = predictor.predictLocal(imagePath, topN)
    val clses = predictResult.clsWithCredits
      .map(_.className).toList.asJava
    val credits = predictResult.clsWithCredits
      .map(_.credit).toList.asJava
    Array(clses, credits)
  }

  def predictDistributed(predictor: Predictor, paths : JavaRDD[String], topN : Int) :
    RDD[Array[JList[String]]] = {
    val sqlContext = SQLContext.getOrCreate(paths.sparkContext)
    val res = predictor.predictDistributed(paths.rdd, topN).map(predictResult => {
      val infor = predictResult.info
      val clsWithCredits = predictResult.clsWithCredits
      val inforList = new JArrayList[String]()
      inforList.add(infor)
      val clses = predictResult.clsWithCredits
        .map(_.className).toList.asJava
      val credits = predictResult.clsWithCredits
        .map(_.credit.toString).toList.asJava
      Array(inforList, clses, credits)
    })
    res
  }
}
