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

package com.intel.analytics.zoo.transform.vision.image3d

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.numeric._
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.dataset.Transformer

import scala.collection.{Iterator, mutable}
import scala.reflect.ClassTag

class Image3D extends Serializable {
  def this(floats: Array[Float], label: Any, path: String) {
    this
    state(Image3D.floats) = floats
    if (null != path) {
      state(Image3D.path) = path
    }
    if (null != label) {
      state(Image3D.label) = label
    }
  }

  private val state = new mutable.HashMap[String, Any]()

  var isValid = true

  def apply[T](key: String): T = state(key).asInstanceOf[T]

  def update(key: String, value: Any): Unit = state(key) = value

  def contains(key: String): Boolean = state.contains(key)

  def hasLabel(): Boolean = state.contains(Image3D.label)

  def getFloats(): Array[Float] = {
    if (state.contains(Image3D.floats)) {
      state(Image3D.floats).asInstanceOf[Array[Float]]
    } else {
      null.asInstanceOf[Array[Float]]
    }
  }

  def getDepth(): Int = {
    if (state.contains(Image3D.width)) {
      state(Image3D.depth).asInstanceOf[Int]
    } else {
      null.asInstanceOf[Int]
    }
  }

  def getWidth(): Int = {
    if (state.contains(Image3D.width)) {
      state(Image3D.width).asInstanceOf[Int]
    } else {
      null.asInstanceOf[Int]
    }
  }

  def getHeight(): Int = {
    if (state.contains(Image3D.width)) {
      state(Image3D.height).asInstanceOf[Int]
    } else {
      null.asInstanceOf[Int]
    }
  }

  def getPath(): String = {
    if (state.contains(Image3D.path)) {
      state(Image3D.path).asInstanceOf[String]
    } else {
      null.asInstanceOf[String]
    }
  }

  def getLabel[T: ClassTag]: T = {
    if (hasLabel()) this (Image3D.label).asInstanceOf[T] else null.asInstanceOf[T]
  }

  def clear(): Unit = {
    state.clear()
  }

  def copyTo(storage: Array[Float], offset: Int,
             toRGB: Boolean = true): Unit = {
    require(contains(Image3D.floats), "there should be floats in Image3D")
    val data = getFloats()
    require(data.length >= getDepth() * getWidth() * getHeight() * 3,
      "float array length should be larger than 3 * ${getDepth()} * ${getWidth()} * ${getHeight()}")
    val frameLength = getDepth() * getWidth() * getHeight()
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
}

object Image3D {
  val label = "label"
  val path = "path"
  val bytes = "bytes"
  val floats = "floats"
  val depth = "depth"
  val width = "width"
  val height = "height"
  // original image width
  val originalW = "originalW"
  val originalH = "originalH"

  def apply(data: Array[Float], path: String = null, label: Any = null)
  : Image3D = new Image3D(data, label, path)

  def apply(): Image3D = new Image3D()
}

abstract class FeatureTransformer() extends Transformer[Image3D, Image3D] {

  def transformTensor(tensor: Tensor[Float]): Tensor[Float] = {tensor}

  def transform(feature: Image3D): Image3D = {
    try {
      // change image to tensor
      val data = feature.getFloats()
      val sizes = Array(feature.getDepth(), feature.getHeight(), feature.getWidth())
      val tensor = Tensor[Float]()
      tensor.set(Storage[Float](data), sizes = sizes)
      val out = transformTensor(tensor).clone()
      feature(Image3D.floats) = out.storage().array()
      feature(Image3D.depth) = out.size(1)
      feature(Image3D.height) = out.size(2)
      feature(Image3D.width) = out.size(3)
//      val result = new Image3D(out.storage().array(), feature.getLabel, feature.getPath)
    } catch {
      case e: Exception =>
        e.printStackTrace()
        feature.isValid = false
    }
    feature
  }

  override def apply(prev: Iterator[Image3D]): Iterator[Image3D] = {
    prev.map(image => {
      transform(image)
      image
    })
  }

  // scalastyle:off methodName
  // scalastyle:off noSpaceBeforeLeftBracket
  def -> [C](other: FeatureTransformer): FeatureTransformer = {
    new ChainedFeatureTransformer(this, other)
  }
}

class ChainedFeatureTransformer(first: FeatureTransformer, last: FeatureTransformer) extends
  FeatureTransformer {
  override def apply(prev: Iterator[Image3D]): Iterator[Image3D] = {
    last(first(prev))
  }

  override def transform(prev: Image3D): Image3D = {
    last.transform(first.transform(prev))
  }
}


class RandomTransformer(transformer: FeatureTransformer, maxProb: Double)
  extends FeatureTransformer {
  override def transform(prev: Image3D): Image3D = {
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
