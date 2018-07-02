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
package com.intel.analytics.zoo.feature

import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.transform.vision.image.ImageFeature
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.feature.common.{BigDLAdapter, Preprocessing}
import com.intel.analytics.zoo.feature.image.{ImageMatToTensor, ImageResize, ImageSet}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}


class ImageFeatureSpec extends FlatSpec with Matchers with BeforeAndAfter {

  "BigDLAdapter" should "adapt BigDL Transformer" in {
    val newResize = BigDLAdapter(ImageResize(1, 1))
    assert(newResize.isInstanceOf[Preprocessing[_, _]])
  }

  "ImageMatToTensor" should "work with both NCHW and NHWC" in {
    val resource = getClass.getClassLoader.getResource("pascal/")
    val data = ImageSet.read(resource.getFile)
    val nhwc = (data -> ImageMatToTensor[Float](format = DataFormat.NHWC)).toLocal()
      .array.head.apply[Tensor[Float]](ImageFeature.imageTensor)
    require(nhwc.isContiguous() == true)

    val data2 = ImageSet.read(resource.getFile)
    require(data2.toLocal().array.head.apply[Tensor[Float]](ImageFeature.imageTensor) == null)
    val nchw = (data2 -> ImageMatToTensor[Float]()).toLocal()
      .array.head.apply[Tensor[Float]](ImageFeature.imageTensor)

    require(nchw.transpose(1, 2).transpose(2, 3).contiguous().storage().array().deep
      == nhwc.storage().array().deep)
  }
}
