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

import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.numeric._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.dataset.{ChainedTransformer, Transformer}

import scala.collection.{Iterator, mutable}
import scala.reflect.ClassTag

class Image3D extends Serializable {
  def this(data: Tensor[Float], label: Any, path: String) {
    this
    state(Image3D.data) = data
    if (null != path) {
      state(Image3D.path) = path
    }
    if (null != label) {
      state(Image3D.label) = label
    }
    state(Image3D.depth) = data.size(1)
    state(Image3D.height) = data.size(2)
    state(Image3D.width) = data.size(3)
  }

  private val state = new mutable.HashMap[String, Any]()

  var isValid = true

  def apply[T](key: String): T = state(key).asInstanceOf[T]

  def update(key: String, value: Any): Unit = state(key) = value

  def contains(key: String): Boolean = state.contains(key)

  def hasLabel(): Boolean = state.contains(Image3D.label)

  def getData(): Tensor[Float] = {
    if (state.contains(Image3D.data)) {
      state(Image3D.data).asInstanceOf[Tensor[Float]]
    } else {
      null.asInstanceOf[Tensor[Float]]
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
}



object Image3D {
  val label = "label"
  val path = "path"
  val data = "data"
  val depth = "depth"
  val width = "width"
  val height = "height"

  def apply(data: Tensor[Float], path: String = null, label: Any = null)
  : Image3D = new Image3D(data, label, path)

  def apply(): Image3D = new Image3D()
}

abstract class FeatureTransformer() extends Transformer[Image3D, Image3D] {

  def transformTensor(tensor: Tensor[Float]): Tensor[Float] = {tensor}

  def transform(feature: Image3D): Image3D = {
    try {
      // change image to tensor
      val data = feature.getData()
      val out = transformTensor(data).clone()
      feature(Image3D.data) = out
      feature(Image3D.depth) = out.size(1)
      feature(Image3D.height) = out.size(2)
      feature(Image3D.width) = out.size(3)
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

  // scalastyle:off methodName
  // scalastyle:off noSpaceBeforeLeftBracket
  override def -> [C](other: Transformer[Image3D, C]): Transformer[Image3D, C] = {
    new ChainedTransformer(this, other)
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
