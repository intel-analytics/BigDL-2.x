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
import com.intel.analytics.bigdl.transform.vision.image.ImageFeature
import org.apache.log4j.Logger

import scala.reflect.ClassTag

/**
 * Transforms tensors that map inputKeys and targetKeys to sample
 * @param inputKeys keys that maps inputs (each input should be a tensor)
 * @param targetKeys keys that maps targets (each target should be a tensor)
 * @param sampleKey key to store sample
 */
class ImageSetToSample[T: ClassTag](inputKeys: Array[String] = Array(ImageFeature.imageTensor),
                       targetKeys: Array[String] = Array(ImageFeature.label),
                       sampleKey: String = ImageFeature.sample)(implicit ev: TensorNumeric[T])
  extends ImageProcessing {
  override def apply(prev: Iterator[ImageFeature]): Iterator[ImageFeature] = {
    prev.map(transform(_))
  }

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
        // If an ImageFeature doesn't contain the specified target(s), the result Sample
        // won't contain labels.
        // In this case the same preprocessor for ImageModels can both handle images with labels
        // (for evaluation) or without labels (for inference).
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
        ImageSetToSample.logger.warn(s"The conversion from ImageFeature to Sample fails for $uri")
        feature(ImageFeature.originalSize) = (-1, -1, -1)
        feature.isValid = false
    }
    feature
  }
}

object ImageSetToSample {
  val logger: Logger = Logger.getLogger(getClass)

  def apply[T: ClassTag](inputKeys: Array[String] = Array(ImageFeature.imageTensor),
            targetKeys: Array[String] = Array(ImageFeature.label),
            sampleKey: String = ImageFeature.sample)
            (implicit ev: TensorNumeric[T]): ImageSetToSample[T] =
    new ImageSetToSample(inputKeys, targetKeys, sampleKey)
}
