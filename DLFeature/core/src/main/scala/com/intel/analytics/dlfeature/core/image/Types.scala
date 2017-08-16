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

package com.intel.analytics.dlfeature.core.image

import com.intel.analytics.bigdl.dataset.Transformer
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.dlfeature.core.label.LabelTransformer
import com.intel.analytics.dlfeature.core.util.MatWrapper

import scala.collection.mutable
import scala.reflect.ClassTag

case class ByteImage(var image: Array[Byte] = null, var dataLength: Int = 0,
  var path: String = "", var label: Option[Any] = None) extends Serializable

class Feature extends Serializable {
  private val state = new mutable.HashMap[String, Any]()

  var inKey: String = ""

  def apply[T](key: String): T = state(key).asInstanceOf[T]

  def update(key: String, value: Any): Unit = state(key) = value

  def contains(key: String): Boolean = state.contains(key)

  def inputMat(): MatWrapper = state(inKey).asInstanceOf[MatWrapper]

  def hasLabel(): Boolean = state.contains(Feature.label)

  def getFloats(): Array[Float] = {
    state(Feature.floats).asInstanceOf[Array[Float]]
  }

  def getWidth(): Int = {
    state(Feature.width).asInstanceOf[Int]
  }

  def getHeight(): Int = {
    state(Feature.height).asInstanceOf[Int]
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
  // current image width
  val width = "width"
  val height = "height"
  // original image width
  val originalW = "originalW"
  val originalH = "originalH"
}

abstract class ImageTransformer() extends Transformer[Feature, Feature]
  with LabelTransformer {

  private var _inKey: Option[String] = None
  private var _outKey: Option[String] = None

  def setInKey(key: String): this.type = {
    _inKey = Some(key)
    this
  }

  def setOutKey(key: String): this.type = {
    _outKey = Some(key)
    this
  }

  def getInKey: String = _inKey.get

  def getOutKey: String = _outKey.get

  def transform(input: MatWrapper, output: MatWrapper, feature: Feature): Boolean

  override def apply(prev: Iterator[Feature]): Iterator[Feature] = {
    prev.map(feature => {
      var input: MatWrapper = null
      var output: MatWrapper = null
      val inKey = if (_inKey.isDefined) getInKey else feature.inKey
      val outKey = if (_outKey.isDefined) getOutKey else inKey
      try {
        input = feature(inKey).asInstanceOf[MatWrapper]
        output = if (inKey != outKey) {
          // create a new Mat
          new MatWrapper()
        } else {
          // inplace operation
          input
        }
        if (transform(input, output, feature) && hasLabelTransfomer()) {
          transformLabel(feature)
        }
        feature(outKey) = output
      } catch {
        case e: Exception =>
          e.printStackTrace()
      } finally {
        if (input != output) {
          if (null != input) input.release()
        }
      }
      feature.inKey = outKey
      feature
    })
  }

  protected def randomOperation(operation: ((MatWrapper, MatWrapper) => MatWrapper),
    input: MatWrapper, output: MatWrapper, maxProb: Double): Boolean = {
    val prob = RNG.uniform(0, 1)
    if (prob < maxProb) {
      operation(input, output)
      true
    } else {
      if (input != output) input.copyTo(output)
      false
    }
  }
}

