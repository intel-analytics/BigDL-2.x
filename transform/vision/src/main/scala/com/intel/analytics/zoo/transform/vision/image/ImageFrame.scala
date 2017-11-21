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

package com.intel.analytics.zoo.transform.vision.image

import java.io.File

import com.intel.analytics.bigdl.dataset.Transformer
import org.apache.commons.io.FileUtils
import org.apache.log4j.Logger
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import scala.reflect.ClassTag

trait ImageFrame {

  def transform(transformer: FeatureTransformer): ImageFrame

  // scalastyle:off methodName
  // scalastyle:off noSpaceBeforeLeftBracket
  def -> [C: ClassTag](transformer: FeatureTransformer): ImageFrame = {
    this.transform(transformer)
  }

  def isLocal(): Boolean = this.isInstanceOf[LocalImageFrame]

  def isDistributed(): Boolean = this.isInstanceOf[DistributedImageFrame]
}

object ImageFrame {
  val logger = Logger.getLogger(getClass)

  /**
   * create LocalImageFrame
   * @param data array of ImageFeature
   */
  def array(data: Array[ImageFeature]): LocalImageFrame = {
    new LocalImageFrame(data)
  }

  /**
   * create DistributedImageFrame
   * @param data rdd of ImageFeature
   */
  def rdd(data: RDD[ImageFeature]): DistributedImageFrame = {
    new DistributedImageFrame(data)
  }

  /**
   * Read image as DistributedImageFrame from local file system or HDFS
   *
   * @param path path to read images. Local or HDFS. Wildcard character are supported.
   * @param sc SparkContext
   * @return DistributedImageFrame
   */
  def read(path: String, sc: SparkContext): DistributedImageFrame = {
    val images = sc.binaryFiles(path).map { case (p, stream) =>
      ImageFeature(stream.toArray(), uri = p)
    }.map(BytesToMat.transform)
    ImageFrame.rdd(images)
  }

  /**
   * Read image as LocalImageFrame from local directory
   *
   * @param path local flatten directory with images
   * @return LocalImageFrame
   */
  def read(path: String): LocalImageFrame = {
    val dir = new File(path)
    require(dir.exists(), s"$path not exists!")
    require(dir.isDirectory, s"$path is not directory!")
    val images = dir.listFiles().map { p =>
      ImageFeature(FileUtils.readFileToByteArray(p), uri = p.getAbsolutePath)
    }.map(BytesToMat.transform)
    ImageFrame.array(images)
  }

  /**
   * Read parquet file as DistributedImageFrame
   *
   * @param path Parquet file path
   * @return DistributedImageFrame
   */
  def readParquet(path: String, spark: SparkSession): DistributedImageFrame = {
    val df = spark.sqlContext.read.parquet(path)
    val images = df.rdd.map(row => {
      val uri = row.getAs[String](ImageFeature.uri)
      val image = row.getAs[Array[Byte]](ImageFeature.bytes)
      ImageFeature(image, uri = uri)
    }).map(BytesToMat.transform)
    ImageFrame.rdd(images)
  }

  /**
   * Write images as parquet file
   *
   * @param path path to read images. Local or HDFS. Wildcard character are supported.
   * @param output Parquet file path
   */
  def writeParquet(path: String, output: String, spark: SparkSession): Unit = {
    import spark.implicits._
    val df = spark.sparkContext.binaryFiles(path)
      .map { case (p, stream) =>
        (p, stream.toArray())
      }.toDF(ImageFeature.uri, ImageFeature.bytes)
    df.write.parquet(output)
  }


  implicit def imageFrameToLocal(imageFrame: ImageFrame): LocalImageFrame = {
    imageFrame.asInstanceOf[LocalImageFrame]
  }

  implicit def imageFrameToDist(imageFrame: ImageFrame): DistributedImageFrame = {
    imageFrame.asInstanceOf[DistributedImageFrame]
  }

  implicit def rddToDistributedImageFrame(rdd: RDD[ImageFeature]): DistributedImageFrame = {
    ImageFrame.rdd(rdd)
  }
}

class LocalImageFrame(var array: Array[ImageFeature]) extends ImageFrame {
  def apply(transformer: FeatureTransformer): LocalImageFrame = {
    array = array.map(transformer.transform)
    this
  }

  def toDistributed(sc: SparkContext): DistributedImageFrame = {
    new DistributedImageFrame(sc.parallelize(array))
  }

  override def transform(transformer: FeatureTransformer): ImageFrame = {
    this.apply(transformer)
  }
}

class DistributedImageFrame(var rdd: RDD[ImageFeature]) extends ImageFrame {
  def apply(transformer: FeatureTransformer): DistributedImageFrame = {
    rdd = transformer(rdd)
    this
  }

  override def transform(transformer: FeatureTransformer): ImageFrame = {
    this.apply(transformer)
  }
}
