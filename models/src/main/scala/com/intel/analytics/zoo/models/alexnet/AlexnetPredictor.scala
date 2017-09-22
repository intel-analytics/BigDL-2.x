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

package com.intel.analytics.zoo.models.alexnet

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}

import com.intel.analytics.bigdl.example.loadmodel.AlexNetPreprocessor.imageSize
import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.serializer.ModuleLoader
import com.intel.analytics.zoo.models.{Predictor, Preprocessor}
import com.intel.analytics.zoo.models.dataset.{ImageParam, ModelContext, PredictResult}
import com.intel.analytics.zoo.models.util.MateToTensor
import com.intel.analytics.zoo.transform.vision.image.{BytesToMat, ImageFeature}
import com.intel.analytics.zoo.transform.vision.image.augmentation.{CenterCrop, PixelNormalizer, Resize}
import org.apache.commons.io.FileUtils
import org.apache.spark.rdd.RDD


class AlexnetPredictor(modelPath : String, meanPath : String) extends Predictor{

  val model : AbstractModule[Activity, Activity, Float] = ModuleLoader.
    loadFromFile[Float](modelPath).evaluate()

  val mean : Tensor[Float] = createMean(meanPath)

  val preprocessor =  AlexnetPreprocessor(mean)

  override def predictLocal(context: ModelContext,
                            preprocessor: Preprocessor = preprocessor): Array[PredictResult] = {

    preprocessor.preprocess(context)

    val input = context.get(ImageParam.tensorInput.toString).asInstanceOf[Tensor[Float]]
    val topNum = context.get(ImageParam.topN.toString).asInstanceOf[Int]
    val res = model.forward(input).asInstanceOf[Tensor[Float]]

    topN(res, topNum)
  }

  override def predict(context: ModelContext, preprocessor: Preprocessor): RDD[Array[PredictResult]] = {

    val batches = context.get(ImageParam.paths.toString).asInstanceOf[RDD[Tensor[Float]]]
    val broadcastModel = ModelBroadcast().broadcast(batches.sparkContext, model)
    null
  }

   def predict(path: String, topNum: Int, preprocessor : Preprocessor): Array[PredictResult] = {
    val bytes = FileUtils.readFileToByteArray(new java.io.File(path.toString))
    val feature = ImageFeature()
    feature(ImageFeature.bytes) = bytes
    // val transformedFeature = preprocessor.transform(feature)
    val input = MateToTensor.bGRToFloatTensor(feature, false)
    val res = model.forward(input).asInstanceOf[Tensor[Float]]
    topN(res, topNum)
  }

  private def createMean(meanFile : String) : Tensor[Float] = {
    val lines = Files.readAllLines(Paths.get(meanFile), StandardCharsets.UTF_8)
    val array = new Array[Float](lines.size)
    lines.toArray.zipWithIndex.foreach {
      x => {
        array(x._2) = x._1.toString.toFloat
      }
    }
    Tensor[Float](array, Array(array.length))
  }
}

class AlexnetPreprocessor(mean : Tensor[Float]) extends Preprocessor {

  val transformer =  BytesToMat() -> Resize(256 , 256) ->
    new PixelNormalizer(mean) -> CenterCrop(imageSize, imageSize)

  override def preprocess(context: ModelContext): Unit = {
    val rawImg = context.get(ImageParam.rawImg.toString)
    val feature = ImageFeature()
    feature(ImageFeature.bytes) = rawImg
    val transformedInput = transformer.transform(feature)
    val input = MateToTensor.bGRToFloatTensor(transformedInput, false)
    context.put(ImageParam.tensorInput.toString, input)
  }
}
object AlexnetPreprocessor {
  def apply(mean: Tensor[Float]): AlexnetPreprocessor = new AlexnetPreprocessor(mean)
}

object AlexnetPredictor{

  def apply(modelPath: String, meanPath : String): AlexnetPredictor = new AlexnetPredictor(modelPath, meanPath)

  def main(args: Array[String]): Unit = {

    val predictor = AlexnetPredictor("/home/jerry/lab/data/bigdl/alexnet.bigdl", "/home/jerry/Downloads/mean.txt")
    val bytes = FileUtils.readFileToByteArray(new java.io.File("/home/jerry/Downloads/cat.jpeg"))
    val context = ModelContext()
    context.put(ImageParam.rawImg.toString, bytes)
    context.put(ImageParam.topN.toString, 2)

    val res = predictor.predictLocal(context)

    println()

  }
}
