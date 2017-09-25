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
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.bigdl.utils.serializer.ModuleLoader
import com.intel.analytics.zoo.models.{Predictor, Preprocessor}
import com.intel.analytics.zoo.models.dataset._
import com.intel.analytics.zoo.transform.vision.image.{BytesToMat, ImageFeature}
import com.intel.analytics.zoo.transform.vision.image.augmentation.{CenterCrop, PixelNormalizer, Resize}
import org.apache.commons.io.FileUtils
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD

@SerialVersionUID(6044110396967995592L)
class AlexnetPredictor(modelPath : String, meanPath : String) extends Predictor with Serializable{

  val model : AbstractModule[Activity, Activity, Float] = ModuleLoader.
    loadFromFile[Float](modelPath).evaluate()

  val mean : Tensor[Float] = createMean(meanPath)

 // val preprocessor =  AlexnetPreprocessor(mean)

  val transformer = ImageToMate() -> BytesToMat() -> Resize(256 , 256) ->
    PixelNormalizer(mean) -> CenterCrop(imageSize, imageSize) -> MateToSample(false)

  private val batchPerPartition = 4

  override def predictLocal(path : String, topNum : Int, preprocessor: Preprocessor = null)
  : Array[PredictResult] = {
    val sample = preprocessor.preprocess(Seq(path).iterator).next()
    val input = sample.input.view(Array(1) ++ sample.input.size())
    val res = model.forward(input).asInstanceOf[Tensor[Float]]
    topN(res, topNum)
  }

  override def predictDistributed(paths : RDD[String], topNum : Int, preprocessor: Preprocessor = null):
    RDD[Array[PredictResult]] = {
    val partitionNum = paths.partitions.length
    val totalBatch = batchPerPartition * partitionNum
    val broadcastModel = ModelBroadcast[Float]().broadcast(paths.sparkContext, model)
    val broadcastProcessor = paths.sparkContext.broadcast(SampleToBatch(
      totalBatch = totalBatch, partitionNum = Some(partitionNum)))
    val predictDataSet = paths.map(path => transformer(Seq(path).iterator))
    val predictResult = predictDataSet.mapPartitions(partition => {
      val localModel = broadcastModel.value()
      val localProcessor = broadcastProcessor.value.cloneTransformer()
      val minibatch = localProcessor(partition.next())
      minibatch.map(batch => {
        val result = localModel.forward(batch.input).asInstanceOf[Tensor[Float]]
        topN(result, topNum)
      })
    })
    predictResult
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

object AlexnetPredictor{

  def apply(modelPath: String, meanPath : String): AlexnetPredictor = new AlexnetPredictor(modelPath, meanPath)

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setMaster("local[1]").setAppName("Test distributed predict")
    .set("spark.shuffle.reduceLocality.enabled", "false")
      .set("spark.shuffle.blockTransferService", "nio")
      .set("spark.scheduler.minRegisteredResourcesRatio", "1.0")
    val sc = new SparkContext(conf)
    Engine.init
    val predictor = AlexnetPredictor("/home/jerry/lab/data/bigdl/alexnet.bigdl", "/home/jerry/Downloads/mean.txt")
    //  val res = predictor.predictLocal("/home/jerry/Downloads/cat.jpeg", 2)
    val paths = sc.parallelize(Array("/home/jerry/Downloads/cat.jpeg"))

    val res = predictor.predictDistributed(paths, 2).collect()
    println()
  }
}
