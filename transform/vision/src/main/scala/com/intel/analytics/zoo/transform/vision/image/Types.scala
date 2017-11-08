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
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.zoo.transform.vision.image.opencv.OpenCVMat
import org.apache.commons.io.FileUtils
import org.apache.log4j.Logger
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession

import scala.collection.{Set, mutable}
import scala.reflect.ClassTag

class ImageFeature extends Serializable {

  import ImageFeature.logger

  /**
   * Create ImageFeature
   *
   * @param bytes image file in bytes
   * @param label label
   * @param uri image uri
   */
  def this(bytes: Array[Byte], label: Any = null, uri: String = null) {
    this
    state(ImageFeature.bytes) = bytes
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

  def bytes(): Array[Byte] = apply[Array[Byte]](ImageFeature.bytes)

  def uri(): String = apply[String](ImageFeature.uri)


  def getFloats(key: String = ImageFeature.floats): Array[Float] = {
    state(key).asInstanceOf[Array[Float]]
  }

  /**
   * get current image size in (height, width, channel)
   *
   * @return (height, width, channel)
   */
  def getSize: (Int, Int, Int) = {
    val mat = opencvMat()
    if (!mat.isReleased) {
      mat.shape()
    } else if (contains(ImageFeature.size)) {
      apply[(Int, Int, Int)](ImageFeature.size)
    } else {
      getOriginalSize
    }
  }

  def getHeight(): Int = getSize._1

  def getWidth(): Int = getSize._2

  def getChannel(): Int = getSize._3

  /**
   * get original image size in (height, width, channel)
   *
   * @return (height, width, channel)
   */
  def getOriginalSize: (Int, Int, Int) = {
    if (contains(ImageFeature.originalSize)) {
      apply[(Int, Int, Int)](ImageFeature.originalSize)
    } else {
      logger.warn("there is no original size stored")
      (-1, -1, -1)
    }
  }

  def getOriginalWidth: Int = getOriginalSize._2

  def getOriginalHeight: Int = getOriginalSize._1

  def getLabel[T: ClassTag]: T = {
    if (hasLabel()) this (ImageFeature.label).asInstanceOf[T] else null.asInstanceOf[T]
  }

  /**
   * imInfo is a tensor that contains height, width, scaleInHeight, scaleInWidth
   * e.g. it is used in SSD and Faster-RCNN to post process the roi detection
   */
  def getImInfo(): Tensor[Float] = {
    val (height, width, _) = getSize
    val (oh, ow, _) = getOriginalSize
    Tensor[Float](T(height, width, height.toFloat / oh, width.toFloat / ow))
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
      s"float array length should be larger than 3 * ${getWidth()} * ${getHeight()}")
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
  // OpenCV mat
  val mat = "mat"
  // image file in bytes
  val bytes = "bytes"
  // image pixels in float array
  val floats = "floats"
  // current image size
  val size = "size"
  // original image size
  val originalSize = "originalSize"
  val cropBbox = "cropBbox"
  val expandBbox = "expandBbox"

  /**
   * Create ImageFeature
   *
   * @param bytes image file in bytes
   * @param label label
   * @param uri image uri
   */
  def apply(bytes: Array[Byte], label: Any = null, uri: String = null)
  : ImageFeature = new ImageFeature(bytes, label, uri)

  def apply(): ImageFeature = new ImageFeature()

  def fromOpenCVMat(openCVMat: OpenCVMat): ImageFeature = {
    val imageFeature = new ImageFeature()
    imageFeature(ImageFeature.mat) = openCVMat
    imageFeature(ImageFeature.size) = openCVMat.shape()
    imageFeature
  }

  val logger = Logger.getLogger(getClass)
}

abstract class FeatureTransformer() extends Transformer[ImageFeature, ImageFeature] {

  import FeatureTransformer.logger

  private var outKey: Option[String] = None

  def setOutKey(key: String): this.type = {
    outKey = Some(key)
    this
  }

  protected def transformMat(feature: ImageFeature): Unit = {}

  def transform(feature: ImageFeature): ImageFeature = {
    if (!feature.isValid) return feature
    try {
      transformMat(feature)
      if (outKey.isDefined) {
        require(outKey.get != ImageFeature.mat, s"the output key should not equal to" +
          s" ${ImageFeature.mat}, please give another name")
        if (feature.contains(outKey.get)) {
          val mat = feature(outKey.get).asInstanceOf[OpenCVMat]
          feature.opencvMat().copyTo(mat)
        } else {
          feature(outKey.get) = feature.opencvMat().clone()
        }
      }
    } catch {
      case e: Exception =>
        val path = if (feature.contains(ImageFeature.uri)) feature(ImageFeature.uri) else ""
        logger.warn(s"failed ${path} in transformer ${getClass}")
        e.printStackTrace()
        feature.isValid = false
    }
    feature
  }

  override def apply(prev: Iterator[ImageFeature]): Iterator[ImageFeature] = {
    prev.map(transform)
  }

  // scalastyle:off methodName
  // scalastyle:off noSpaceBeforeLeftBracket
  def ->(other: FeatureTransformer): FeatureTransformer = {
    new ChainedFeatureTransformer(this, other)
  }

  // scalastyle:off methodName
  // scalastyle:off noSpaceBeforeLeftBracket
  override def ->[C](other: Transformer[ImageFeature, C]): Transformer[ImageFeature, C] = {
    new ChainedTransformer(this, other)
  }
}

object FeatureTransformer {
  val logger = Logger.getLogger(getClass)
}

class ChainedFeatureTransformer(first: FeatureTransformer, last: FeatureTransformer) extends
  FeatureTransformer {

  override def transform(prev: ImageFeature): ImageFeature = {
    last.transform(first.transform(prev))
  }
}


class RandomTransformer(transformer: FeatureTransformer, maxProb: Double)
  extends FeatureTransformer {
  override def transform(prev: ImageFeature): ImageFeature = {
    if (RNG.uniform(0, 1) < maxProb) {
      transformer.transform(prev)
    }
    prev
  }

  override def toString: String = {
    s"Random[${transformer.getClass.getCanonicalName}, $maxProb]"
  }
}

object RandomTransformer {
  def apply(transformer: FeatureTransformer, maxProb: Double): RandomTransformer =
    new RandomTransformer(transformer, maxProb)
}

object Image {
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
    }
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
    val images = dir.listFiles().map { p =>
      ImageFeature(FileUtils.readFileToByteArray(p), uri = p.getAbsolutePath)
    }
    ImageFrame.array(images)
  }

  /**
   * Read sequence file as DistributedImageFrame
   *
   * @param path sequence file path
   * @return DistributedImageFrame
   */
  def readSequenceFile(path: String, spark: SparkSession): DistributedImageFrame = {
    val df = spark.sqlContext.read.parquet(path)
    val images = df.rdd.map(row => {
      val uri = row.getAs[String](ImageFeature.uri)
      val image = row.getAs[Array[Byte]](ImageFeature.bytes)
      ImageFeature(image, uri = uri)
    })
    ImageFrame.rdd(images)
  }

  /**
   * Write images as sequence file
   *
   * @param path path to read images. Local or HDFS. Wildcard character are supported.
   * @param output sequence file path
   */
  def writeSequenceFile(path: String, output: String, spark: SparkSession): Unit = {
    import spark.implicits._
    val df = spark.sparkContext.binaryFiles(path)
      .map { case (p, stream) =>
        (p, stream.toArray())
      }.toDF(ImageFeature.uri, ImageFeature.bytes)
    df.write.parquet(output)
  }
}
