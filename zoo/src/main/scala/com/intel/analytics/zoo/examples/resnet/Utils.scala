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

package com.intel.analytics.zoo.examples.resnet

import java.nio.ByteBuffer

import com.intel.analytics.bigdl.dataset.image.{CropCenter, CropRandom}
import com.intel.analytics.bigdl.dataset.{ByteRecord, MiniBatch, SampleToMiniBatch}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.ImageFeature
import com.intel.analytics.bigdl.transform.vision.image.augmentation.RandomAlterAspect
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.zoo.feature.FeatureSet
import com.intel.analytics.zoo.feature.image._
import com.intel.analytics.zoo.feature.pmem.{DRAM, MemoryType}
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.EngineRef
import org.apache.hadoop.io.Text
import org.apache.spark.SparkContext
import scopt.OptionParser

object Utils {
  case class TrainParams(
    folder: String = "./",
    checkpoint: Option[String] = None,
    modelSnapshot: Option[String] = None,
    stateSnapshot: Option[String] = None,
    optnet: Boolean = false,
    depth: Int = 20,
    classes: Int = 10,
    shortcutType: String = "A",
    batchSize: Int = 128,
    nepochs: Int = 165,
    learningRate: Double = 0.1,
    weightDecay: Double = 1e-4,
    momentum: Double = 0.9,
    dampening: Double = 0.0,
    nesterov: Boolean = true,
    graphModel: Boolean = false,
    warmupEpoch: Int = 0,
    maxLr: Double = 0.0,
    memoryType: String = "dram")

  val trainParser = new OptionParser[TrainParams]("BigDL ResNet Example") {
    head("Train ResNet model on single node")
    opt[String]('f', "folder")
      .text("where you put your training files")
      .action((x, c) => c.copy(folder = x))
    opt[String]("model")
      .text("model snapshot location")
      .action((x, c) => c.copy(modelSnapshot = Some(x)))
    opt[String]("state")
      .text("state snapshot location")
      .action((x, c) => c.copy(stateSnapshot = Some(x)))
    opt[String]("cache")
      .text("where to cache the model")
      .action((x, c) => c.copy(checkpoint = Some(x)))
    opt[Boolean]("optnet")
      .text("shared gradients and caches to reduce memory usage")
      .action((x, c) => c.copy(optnet = x))
    opt[Int]("depth")
      .text("depth of ResNet, 18 | 20 | 34 | 50 | 101 | 152 | 200")
      .action((x, c) => c.copy(depth = x))
    opt[Int]("classes")
      .text("classes of ResNet")
      .action((x, c) => c.copy(classes = x))
    opt[String]("shortcutType")
      .text("shortcutType of ResNet, A | B | C")
      .action((x, c) => c.copy(shortcutType = x))
    opt[Int]("batchSize")
      .text("batchSize of ResNet, 64 | 128 | 256 | ..")
      .action((x, c) => c.copy(batchSize = x))
    opt[Int]("nEpochs")
      .text("number of epochs of ResNet; default is 165")
      .action((x, c) => c.copy(nepochs = x))
    opt[Double]("learningRate")
      .text("initial learning rate of ResNet; default is 0.1")
      .action((x, c) => c.copy(learningRate = x))
    opt[Double]("momentum")
      .text("momentum of ResNet; default is 0.9")
      .action((x, c) => c.copy(momentum = x))
    opt[Double]("weightDecay")
      .text("weightDecay of ResNet; default is 1e-4")
      .action((x, c) => c.copy(weightDecay = x))
    opt[Double]("dampening")
      .text("dampening of ResNet; default is 0.0")
      .action((x, c) => c.copy(dampening = x))
    opt[Boolean]("nesterov")
      .text("nesterov of ResNet; default is trye")
      .action((x, c) => c.copy(nesterov = x))
    opt[Unit]('g', "graphModel")
      .text("use graph model")
      .action((x, c) => c.copy(graphModel = true))
    opt[Int]("warmupEpoch")
      .text("warmup epoch")
      .action((x, c) => c.copy(warmupEpoch = x))
    opt[Double]("maxLr")
      .text("maxLr")
      .action((x, c) => c.copy(maxLr = x))
    opt[String]("memoryType")
      .text("what memory type will it runs on")
      .action((x, c) => c.copy(memoryType = x))
  }

  def readLabel(data: Text): String = {
    val dataArr = data.toString.split("\n")
    if (dataArr.length == 1) {
      dataArr(0)
    } else {
      dataArr(1)
    }
  }

  def readFromSeqFiles(
                        url: String, sc: SparkContext, classNum: Int) = {
    val nodeNumber = EngineRef.getNodeNumber()
    val coreNumber = EngineRef.getCoreNumber()
    val rawData = sc.sequenceFile(url, classOf[Text], classOf[Text],
      nodeNumber * coreNumber).map(image => {
      ByteRecord(image._2.copyBytes(), readLabel(image._1).toFloat)
    }).filter(_.label <= classNum)
    rawData
  }

  def byteRecordToImageFeature(record: ByteRecord): ImageFeature = {
    val rawBytes = record.data
    val label = Tensor[Float](T(record.label))
    val imgBuffer = ByteBuffer.wrap(rawBytes)
    val width = imgBuffer.getInt
    val height = imgBuffer.getInt
    val bytes = new Array[Byte](3 * width * height)
    System.arraycopy(imgBuffer.array(), 8, bytes, 0, bytes.length)
    val imf = ImageFeature(bytes, label)
    imf(ImageFeature.originalSize) = (height, width, 3)
    imf
  }

  def loadImageNetTrainDataSet(path: String, sc: SparkContext, imageSize: Int, batchSize: Int,
                       classNum: Int=1000, memoryType: MemoryType =DRAM)
  : FeatureSet[MiniBatch[Float]] = {
    val rawData = readFromSeqFiles(path, sc, classNum)
      .map(byteRecordToImageFeature(_))
      .setName("ImageNet2012 Training Set")

    val featureSet = FeatureSet.rdd(rawData, memoryType = memoryType)
    val transformer = ImagePixelBytesToMat() ->
      RandomAlterAspect() ->
      ImageRandomCropper(224, 224, true, CropRandom) ->
      ImageChannelScaledNormalizer(104, 117, 123, 0.0078125) ->
      ImageMatToTensor[Float](toRGB = false) ->
      ImageSetToSample[Float](inputKeys = Array(ImageFeature.imageTensor),
        targetKeys = Array(ImageFeature.label)) ->
      ImageFeatureToSample[Float]() ->
      SampleToMiniBatch[Float](batchSize)
    featureSet.transform(transformer)
  }

  def loadImageNetValDataSet(path: String, sc: SparkContext, imageSize: Int, batchSize: Int,
                     classNum: Int = 1000, memoryType: MemoryType = DRAM)
  : FeatureSet[MiniBatch[Float]] = {
    val rawData = readFromSeqFiles(path, sc, classNum)
      .map(byteRecordToImageFeature(_))
      .setName("ImageNet2012 Val Set")

    val featureSet = FeatureSet.rdd(rawData, memoryType = memoryType)
    val transformer = ImagePixelBytesToMat() ->
      ImageRandomResize(256, 256) ->
      ImageRandomCropper(224, 224, false, CropCenter) ->
      ImageChannelScaledNormalizer(104, 117, 123, 0.0078125) ->
      ImageMatToTensor[Float](toRGB = false) ->
      ImageSetToSample[Float](inputKeys = Array(ImageFeature.imageTensor),
        targetKeys = Array(ImageFeature.label)) ->
      ImageFeatureToSample[Float]() ->
      SampleToMiniBatch[Float](batchSize)
    featureSet.transform(transformer)
  }

}
