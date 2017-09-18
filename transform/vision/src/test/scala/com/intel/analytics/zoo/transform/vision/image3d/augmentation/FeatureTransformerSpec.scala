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

package com.intel.analytics.zoo.transform.vision.image3d.augmentation

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.zoo.transform.vision.image3d.Image3D
import org.scalatest.{FlatSpec, Matchers}

class FeatureTransformerSpec extends FlatSpec with Matchers{
  "FeatureTransformer" should "work properly" in {val seed = 100
    RNG.setSeed(seed)
    val input = Tensor[Float](60, 70, 80)
    input.apply1(e => RNG.uniform(0, 1).toFloat)
    val image = Image3D(input.storage().array())
    image(Image3D.depth) = 60
    image(Image3D.height) = 70
    image(Image3D.width) = 80
    val start = Array[Int](10, 20, 20)
    val patchSize = Array[Int](20, 30, 30)
    val rotAngles = Array[Double](0, 0, math.Pi/3.7)
    val transformer = Crop(start, patchSize) -> Rotate(rotAngles)
    val out = transformer.transform(image)
    assert(out.getDepth() == 20)
    assert(out.getHeight() == 30)
    assert(out.getWidth() == 30)
  }

}
