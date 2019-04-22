/*
 * Copyright 2018 Analytics Zoo Authors.
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
package com.intel.analytics.zoo.feature.image

import com.intel.analytics.bigdl.dataset.ArraySample
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image.{FeatureTransformer, ImageFeature}
import org.apache.log4j.Logger

import scala.reflect.ClassTag

/**
 * Transforms tensors that map inputKeys and targetKeys to sample
 * @param inputKeys keys that maps inputs (each input should be a tensor)
 * @param targetKeys keys that maps targets (each target should be a tensor)
 * @param sampleKey key to store sample
 */
class ImageSetToSample[T: ClassTag](inputKeys: Array[String] = Array(ImageFeature.imageTensor),
                       targetKeys: Array[String] = null,
                       sampleKey: String = ImageFeature.sample)(implicit ev: TensorNumeric[T])
  extends ImageProcessing {
  private val internalTransformer = InternalImageFrameToSample[T](inputKeys, targetKeys, sampleKey)
  override def apply(prev: Iterator[ImageFeature]): Iterator[ImageFeature] = {
    prev.map(transform(_))
  }

  override def transform(feature: ImageFeature): ImageFeature = {
    internalTransformer.transform(feature)
  }
}

object ImageSetToSample {
  def apply[T: ClassTag](inputKeys: Array[String] = Array(ImageFeature.imageTensor),
            targetKeys: Array[String] = null,
            sampleKey: String = ImageFeature.sample)
            (implicit ev: TensorNumeric[T]): ImageSetToSample[T] =
    new ImageSetToSample(inputKeys, targetKeys, sampleKey)
}

class InternalImageFrameToSample[T: ClassTag](
     inputKeys: Array[String] = Array(ImageFeature.imageTensor),
     targetKeys: Array[String] = null,
     sampleKey: String = ImageFeature.sample)
   (implicit ev: TensorNumeric[T]) extends FeatureTransformer {

  override def transform(feature: ImageFeature): ImageFeature = {
    if (!feature.isValid) return feature
    try {
      val inputs = inputKeys.map(key => {
        val input = feature[Tensor[T]](key)
        require(input.isInstanceOf[Tensor[T]], s"the input $key should be tensor")
        input.asInstanceOf[Tensor[T]]
      })
      val sample = if (targetKeys == null) {
        ArraySample[T](inputs)
      } else {
        val targets = targetKeys.flatMap(key => {
          if (feature.contains(key)) {
            val target = feature[Tensor[T]](key)
            require(target.isInstanceOf[Tensor[T]], s"the target $key should be tensor")
            Some(target.asInstanceOf[Tensor[T]])
          }
          else None
        })
        if (targets.length > 0) ArraySample[T](inputs, targets)
        else ArraySample[T](inputs)
      }
      feature(sampleKey) = sample
    } catch {
      case e: Exception =>
        e.printStackTrace()
        val uri = feature.uri()
        InternalImageFrameToSample.logger.warn(s"convert imageframe to sample fail for $uri")
        feature(ImageFeature.originalSize) = (-1, -1, -1)
        feature.isValid = false
    }
    feature
  }
}

object InternalImageFrameToSample {
  val logger = Logger.getLogger(getClass)

  def apply[T: ClassTag](
      inputKeys: Array[String] = Array(ImageFeature.imageTensor),
      targetKeys: Array[String] = null,
      sampleKey: String = ImageFeature.sample)
    (implicit ev: TensorNumeric[T]): InternalImageFrameToSample[T] = {
    new InternalImageFrameToSample[T](inputKeys, targetKeys, sampleKey)
  }
}
