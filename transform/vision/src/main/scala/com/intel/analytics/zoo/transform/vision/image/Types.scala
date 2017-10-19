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

import com.intel.analytics.bigdl.dataset.{ChainedTransformer, Transformer}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.zoo.transform.vision.image.opencv.OpenCVMat
import org.apache.log4j.Logger

import scala.collection.{Iterator, mutable}
import scala.reflect.ClassTag

class ImageFeature extends Serializable {
  import ImageFeature.logger
  def this(bytes: Array[Byte], label: Any = null, path: String = null) {
    this
    state(ImageFeature.bytes) = bytes
    if (null != path) {
      state(ImageFeature.path) = path
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

  def hasLabel(): Boolean = state.contains(ImageFeature.label)

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
  val path = "path"
  val mat = "mat"
  val bytes = "bytes"
  val floats = "floats"
  val width = "width"
  val height = "height"
  // original image width
  val originalW = "originalW"
  val originalH = "originalH"
  val cropBbox = "cropBbox"
  val expandBbox = "expandBbox"

  def apply(bytes: Array[Byte], label: Any = null, path: String = null)
  : ImageFeature = new ImageFeature(bytes, label, path)

  def apply(): ImageFeature = new ImageFeature()

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
        val path = if (feature.contains(ImageFeature.path)) feature(ImageFeature.path) else ""
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
  def -> (other: FeatureTransformer): FeatureTransformer = {
    new ChainedFeatureTransformer(this, other)
  }

  // scalastyle:off methodName
  // scalastyle:off noSpaceBeforeLeftBracket
  override def -> [C](other: Transformer[ImageFeature, C]): Transformer[ImageFeature, C] = {
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
