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

package com.intel.analytics.zoo.pipeline.common

import java.io.File

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.pipeline.common.dataset.LocalByteRoiimageReader
import com.intel.analytics.zoo.pipeline.common.dataset.roiimage._
import com.intel.analytics.zoo.pipeline.fasterrcnn.FrcnnToBatch
import com.intel.analytics.zoo.transform.vision.image.{BytesToMat, MatToFloats}
import com.intel.analytics.zoo.transform.vision.image.augmentation.{AspectScale, ChannelNormalize, Resize}
import org.apache.hadoop.io.Text
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import scala.io.Source


object IOUtils {
  def loadSeqFiles(nPartition: Int, seqFloder: String, sc: SparkContext)
  : (RDD[SSDByteRecord], RDD[String]) = {
    val data = sc.sequenceFile(seqFloder, classOf[Text], classOf[Text],
      nPartition).map(x => SSDByteRecord(x._2.copyBytes(), x._1.toString))
    val paths = data.map(x => x.path)
    (data, paths)
  }

  def loadLocalFolder(nPartition: Int, folder: String, sc: SparkContext)
  : (RDD[SSDByteRecord], RDD[String]) = {
    val roiDataset = localImagePaths(folder).map(RoiImagePath(_))
    val imgReader = LocalByteRoiimageReader()
    val data = sc.parallelize(roiDataset.map(roidb => imgReader.transform(roidb)),
      nPartition)
    (data, data.map(_.path))
  }

  def localImagePaths(folder: String): Array[String] = {
    new File(folder).listFiles().map(_.getAbsolutePath)
  }

  def preprocessSsd(rdd: RDD[SSDByteRecord], resolution: Int, meansRGB: (Float, Float, Float),
    scale: Float, nPartition: Int,
    batchPerPartition: Int = 1): RDD[ImageMiniBatch] = {
    val preProcessor = RecordToFeature() ->
      BytesToMat() ->
      Resize(resolution, resolution) ->
      ChannelNormalize(meansRGB._1, meansRGB._2, meansRGB._3, scale, scale, scale) ->
      MatToFloats(validHeight = resolution, validWidth = resolution) ->
      RoiImageToBatch(nPartition * batchPerPartition, false, Some(nPartition))
    preProcessor(rdd).asInstanceOf[RDD[ImageMiniBatch]]
  }

  def preprocessSsdVgg(rdd: RDD[SSDByteRecord], resolution: Int, nPartition: Int,
    batchPerPartition: Int = 1): RDD[ImageMiniBatch] = {
    preprocessSsd(rdd, resolution, (123f, 117f, 104f), 1f, nPartition, batchPerPartition)
  }

  def preprocessSsdMobilenet(rdd: RDD[SSDByteRecord], resolution: Int, nPartition: Int,
    batchPerPartition: Int = 1): RDD[ImageMiniBatch] = {
    preprocessSsd(rdd, resolution, (127.5f, 127.5f, 127.5f), 1 / 0.007843f,
      nPartition, batchPerPartition)
  }

  def preprocessFrcnn(rdd: RDD[SSDByteRecord], resolution: Int,
    scaleMultipleOf: Int, nPartition: Int): RDD[ImageMiniBatch] = {
    val preProcessor = RecordToFeature() ->
      BytesToMat() ->
      AspectScale(resolution, scaleMultipleOf) ->
      MatToFloats(validHeight = 100, 100, meanRGB = Some(122.7717f, 115.9465f, 102.9801f)) ->
      FrcnnToBatch(nPartition, false, Some(nPartition))
    preProcessor(rdd).asInstanceOf[RDD[ImageMiniBatch]]
  }

  def preprocessFrcnnVgg(rdd: RDD[SSDByteRecord], nPartition: Int): RDD[ImageMiniBatch] = {
    preprocessFrcnn(rdd, 600, 1, nPartition)
  }

  def preprocessFrcnnPvanet(rdd: RDD[SSDByteRecord], nPartition: Int): RDD[ImageMiniBatch] = {
    preprocessFrcnn(rdd, 640, 32, nPartition)
  }

  def loadClasses(fileName: String): Array[String] = {
    Source.fromFile(fileName).getLines().toArray
  }

  def saveTextResults(output: RDD[Tensor[Float]], paths: RDD[String], path: String): Unit = {
    output.zip(paths).map { case (res: Tensor[Float], path: String) =>
      BboxUtil.resultToString(res, path)
    }.saveAsTextFile(path)
  }

  def visualizeDetections(output: RDD[Tensor[Float]], paths: RDD[String],
    classNames: Array[String], path: String): Unit = {
    paths.zip(output).foreach(pair => {
      val decoded = BboxUtil.decodeRois(pair._2)
      Visualizer.visDetection(pair._1, decoded, classNames, thresh = 0.6f,
        outPath = path)
    })
  }
}

object ModelType {
  val ssd = "ssd"
  val frcnn = "frcnn"
  val vgg = "vgg"
  val pvanet = "pvanet"
  val mobilenet = "mobilenet"
}

