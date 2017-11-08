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

import com.intel.analytics.bigdl.dataset.{ChainedTransformer, Transformer}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.{Engine, T}
import com.intel.analytics.zoo.transform.vision.image.opencv.OpenCVMat
import org.apache.commons.io.FileUtils
import org.apache.log4j.{Logger}
import org.apache.spark.SparkContext
import org.apache.spark.sql.{SparkSession}

import scala.collection.{Iterator, Set, mutable}
import scala.reflect.ClassTag

class ImageFeature extends Serializable {

  import ImageFeature.logger

  def this(image: Array[Byte], label: Any = null, uri: String = null) {
    this
    state(ImageFeature.image) = image
    if (null != uri) {
      state(ImageFeature.uri) = uri
    }
    if (null != label) {
      state(ImageFeature.label) = label
    }
  }

  private val state = new mutable.HashMap[String, Any]()

  var isValid = true


  def apply[T](key: String): T = state(key).asInstanceOf[T]

  def update(key: String, value: Any): Unit = state(key) = value

  def contains(key: String): Boolean = state.contains(key)

  def opencvMat(): OpenCVMat = state(ImageFeature.mat).asInstanceOf[OpenCVMat]

  def keys(): Set[String] = state.keySet

  def hasLabel(): Boolean = state.contains(ImageFeature.label)

  def image: Array[Byte] = apply[Array[Byte]](ImageFeature.image)

  def uri: String = apply[String](ImageFeature.uri)


  def getFloats(key: String = ImageFeature.floats): Array[Float] = {
    state(key).asInstanceOf[Array[Float]]
  }

  def getWidth(): Int = {
    if (state.contains(ImageFeature.width)) state(ImageFeature.width).asInstanceOf[Int]
    else opencvMat().width()
  }

  def getHeight(): Int = {
    if (state.contains(ImageFeature.height)) state(ImageFeature.height).asInstanceOf[Int]
    else opencvMat().height()
  }

  def getOriginalWidth: Int = state(ImageFeature.originalW).asInstanceOf[Int]

  def getOriginalHeight: Int = state(ImageFeature.originalH).asInstanceOf[Int]

  def getLabel[T: ClassTag]: T = {
    if (hasLabel()) this (ImageFeature.label).asInstanceOf[T] else null.asInstanceOf[T]
  }

  def getImInfo(): Tensor[Float] = {
    Tensor[Float](T(getHeight(), getWidth(), getHeight().toFloat / getOriginalHeight,
      getWidth().toFloat / getOriginalWidth))
  }

  def clear(): Unit = {
    state.clear()
    isValid = true
  }


  def copyTo(storage: Array[Float], offset: Int, floatKey: String = ImageFeature.floats,
    toRGB: Boolean = true): Unit = {
    require(contains(floatKey), s"there should be ${floatKey} in ImageFeature")
    val data = getFloats(floatKey)
    require(data.length >= getWidth() * getHeight() * 3,
      "float array length should be larger than 3 * ${getWidth()} * ${getHeight()}")
    val frameLength = getWidth() * getHeight()
    require(frameLength * 3 + offset <= storage.length)
    var j = 0
    if (toRGB) {
      while (j < frameLength) {
        storage(offset + j) = data(j * 3 + 2)
        storage(offset + j + frameLength) = data(j * 3 + 1)
        storage(offset + j + frameLength * 2) = data(j * 3)
        j += 1
      }
    } else {
      while (j < frameLength) {
        storage(offset + j) = data(j * 3)
        storage(offset + j + frameLength) = data(j * 3 + 1)
        storage(offset + j + frameLength * 2) = data(j * 3 + 2)
        j += 1
      }
    }
  }

  /**
   * Convert ImageFeature to image tensor
   *
   * @param floatKey key that maps the float array
   * @param toChw transpose the image from hwc to chw
   * @return tensor that represents an image
   */
  def toTensor(floatKey: String, toChw: Boolean = true): Tensor[Float] = {
    val (floats, size) = if (contains(floatKey)) {
      (getFloats(floatKey),
        Array(getHeight(), getWidth(), 3))
    } else {
      logger.warn(s"please add MatToFloats(out_key = $floatKey) in the end of pipeline if you" +
        s"are transforming an rdd")
      val mat = opencvMat()
      val floats = new Array[Float](mat.height() * mat.width() * 3)
      OpenCVMat.toFloatBuf(mat, floats)
      (floats, Array(mat.height(), mat.width(), 3))
    }
    var image = Tensor(Storage(floats)).resize(size)
    if (toChw) {
      // transpose the shape of image from (h, w, c) to (c, h, w)
      image = image.transpose(1, 3).transpose(2, 3).contiguous()
    }
    image
  }
}

object ImageFeature {
  val label = "label"
  val uri = "uri"
  val mat = "mat"
  // image file in bytes
  val image = "image"
  val floats = "floats"
  val width = "width"
  val height = "height"
  // original image width
  val originalW = "originalW"
  val originalH = "originalH"
  val cropBbox = "cropBbox"
  val expandBbox = "expandBbox"

  def apply(bytes: Array[Byte], label: Any = null, uri: String = null)
  : ImageFeature = new ImageFeature(bytes, label, uri)

  def apply(): ImageFeature = new ImageFeature()

  val logger = Logger.getLogger(getClass)
}

object Image {
  def read(path: String, sc: SparkContext): DistributedImageFrame = {
    val images = sc.binaryFiles(path).map { case (p, stream) =>
      ImageFeature(stream.toArray(), uri = p)
    }
    ImageFrame.rdd(images)
  }

  def read(path: String): LocalImageFrame = {
    val dir = new File(path)
    require(dir.exists(), s"$path not exists!")
    val images = dir.listFiles().map { p =>
      ImageFeature(FileUtils.readFileToByteArray(p), uri = p.getAbsolutePath)
    }
    ImageFrame.array(images)
  }

  def readSequenceFile(path: String, sc: SparkSession): DistributedImageFrame = {
    val df = sc.sqlContext.read.parquet(path)
    val images = df.rdd.map(row => {
      val uri = row.getAs[String](ImageFeature.uri)
      val image = row.getAs[Array[Byte]](ImageFeature.image)
      ImageFeature(image, uri = uri)
    })
    ImageFrame.rdd(images)
  }

  def writeSequenceFile(path: String, output: String, spark: SparkSession): Unit = {
    import spark.implicits._
    val df = spark.sparkContext.binaryFiles(path)
      .map { case (p, stream) =>
        (p, stream.toArray())
      }.toDF(ImageFeature.uri, ImageFeature.image)
    df.write.parquet(output)
  }
}
