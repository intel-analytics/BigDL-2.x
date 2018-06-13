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

import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.ImageFeature
import com.intel.analytics.bigdl.transform.vision.image

import scala.reflect.ClassTag

class ImageMatToTensor[T: ClassTag](
    toRGB: Boolean = false,
    tensorKey: String = ImageFeature.imageTensor,
    shareBuffer: Boolean = true,
    format: DataFormat = DataFormat.NCHW)(implicit ev: TensorNumeric[T])
  extends ImageProcessing {

  private val internalResize = new image.MatToTensor[T](toRGB, tensorKey, shareBuffer)
  override def apply(prev: Iterator[ImageFeature]): Iterator[ImageFeature] = {
    format match {
      case DataFormat.NHWC =>
        prev.map { iter =>
          val imf = transform(iter)
          val tensor = imf[Tensor[T]](tensorKey)
          imf(tensorKey) = tensor.transpose(1, 2).transpose(2, 3)
          imf
        }
      case DataFormat.NCHW => internalResize.apply(prev)
      case other => throw new IllegalArgumentException(s"Unsupported format:" +
        s" $format. Only NCHW and NHWC are supported.")
    }
  }
}

object ImageMatToTensor {
  def apply[T: ClassTag](
      toRGB: Boolean = false,
      tensorKey: String = ImageFeature.imageTensor,
      shareBuffer: Boolean = true,
      format: DataFormat = DataFormat.NCHW
  )(implicit ev: TensorNumeric[T]): ImageMatToTensor[T] =
    new ImageMatToTensor[T](toRGB, tensorKey, shareBuffer, format)
}
