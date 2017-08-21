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

package com.intel.analytics.zoo.feature.core.image

import com.intel.analytics.bigdl.dataset.Transformer
import com.intel.analytics.zoo.feature.core.util.MatWrapper
import scala.collection.{Iterator, mutable}
import scala.reflect.ClassTag

class Feature extends Serializable {
  def this(bytes: Array[Byte], path: Option[String] = None, label: Option[Any] = None) {
    this
    state(Feature.bytes) = bytes
    if (path.isDefined) {
      state(Feature.path) = path.get
    }
    if (label.isDefined) {
      state(Feature.label) = label.get
    }
  }

  private val state = new mutable.HashMap[String, Any]()


  def apply[T](key: String): T = state(key).asInstanceOf[T]

  def update(key: String, value: Any): Unit = state(key) = value

  def contains(key: String): Boolean = state.contains(key)

  def inputMat(): MatWrapper = state(Feature.mat).asInstanceOf[MatWrapper]

  def hasLabel(): Boolean = state.contains(Feature.label)

  def getFloats(): Array[Float] = {
    state(Feature.floats).asInstanceOf[Array[Float]]
  }

  def getWidth(): Int = {
    inputMat().width()
  }

  def getHeight(): Int = {
    inputMat().height()
  }

  def getOriginalWidth: Int = state(Feature.originalW).asInstanceOf[Int]

  def getOriginalHeight: Int = state(Feature.originalH).asInstanceOf[Int]

  def getLabel[T: ClassTag]: T = {
    if (hasLabel()) this (Feature.label).asInstanceOf[T] else null.asInstanceOf[T]
  }

  def clear(): Unit = {
    state.clear()
  }


  def copyTo(storage: Array[Float], offset: Int,
    toRGB: Boolean = true): Unit = {
    val data = getFloats()
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
}

object Feature {
  val label = "label"
  val path = "path"
  val mat = "mat"
  val bytes = "bytes"
  val floats = "floats"
  // original image width
  val originalW = "originalW"
  val originalH = "originalH"

  def apply(bytes: Array[Byte], path: Option[String] = None, label: Option[Any] = None): Feature =
    new Feature(bytes, path, label)

  def apply(): Feature = new Feature()
}

abstract class FeatureTransformer() extends SingleTransformer[Feature, Feature] {
  private var outKey: Option[String] = None

  def setOutKey(key: String): this.type = {
    outKey = Some(key)
    this
  }

  def transform(feature: Feature): Unit

  override def apply(feature: Feature): Feature = {
    try {
      transform(feature)
    } catch {
      case e: Exception =>
        e.printStackTrace()
    }
    if (outKey.isDefined) {
      require(outKey.get != Feature.mat)
      if (feature.contains(outKey.get)) {
        val mat = feature(outKey.get).asInstanceOf[MatWrapper]
        feature.inputMat().copyTo(mat)
      } else {
        feature(outKey.get) = feature.inputMat().clone()
      }
    }
    feature
  }
}

trait SingleTransformer[A, B] extends Serializable {
  def apply(prev: A): B

  // scalastyle:off methodName
  // scalastyle:off noSpaceBeforeLeftBracket
  def -> [C](other: SingleTransformer[B, C]): SingleTransformer[A, C] = {
    new SingleChainedTransformer(this, other)
  }

  def toIterator: Transformer[A, B] = {
    new TransfomerIterator(this)
  }
}


class SingleChainedTransformer[A, B, C]
(first: SingleTransformer[A, B], last: SingleTransformer[B, C])
  extends SingleTransformer[A, C] {
  override def apply(prev: A): C = {
    last(first(prev))
  }
}


class TransfomerIterator[A, B](singleTransformer: SingleTransformer[A, B])
  extends Transformer[A, B] {
  override def apply(prev: Iterator[A]): Iterator[B] = {
    prev.map(singleTransformer(_))
  }
}
