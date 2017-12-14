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

package com.intel.analytics.bigdl.optim

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.models.inception.Inception_v1_NoAuxClassifier
import com.intel.analytics.bigdl.nn.{Sequential, SpatialConvolution, Tanh}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image._
import com.intel.analytics.bigdl.transform.vision.image.augmentation.{CenterCrop, ChannelNormalize, Resize}
import com.intel.analytics.bigdl.utils.{Engine, LocalModule}
import com.intel.analytics.bigdl.utils.RandomGenerator._
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class LocalPredictorSpec extends FlatSpec with Matchers with BeforeAndAfter {

  private val nodeNumber = 1
  private val coreNumber = 4

  before {
    System.setProperty("bigdl.localMode", "true")
    Engine.init(nodeNumber, coreNumber, false)
  }

  after {
    System.clearProperty("bigdl.localMode")
  }

  "predictImage" should "work properly" in {
    import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
    RNG.setSeed(100)
    val resource = getClass.getClassLoader.getResource("pascal/")
    val imageFrame = ImageFrame.read(resource.getFile) ->
      Resize(256, 256) -> CenterCrop(224, 224) ->
      ChannelNormalize(0.485f, 0.456f, 0.406f, 0.229f, 0.224f, 0.225f) ->
      MatToTensor() -> ImageFrameToSample()
    val model = Inception_v1_NoAuxClassifier(classNum = 20)
    val detection = model.predictImage(imageFrame).toLocal()
    val feature = detection.array.head
    println(feature(ImageFeature.predict))

    val imageFeatures = detection.array
    val prob = imageFeatures.map(x => x[Tensor[Float]](ImageFeature.predict))
    val data = imageFeatures.map(_[Sample[Float]](ImageFeature.sample))
    prob(0) should be(model.evaluate().forward(data(0).feature.reshape(Array(1, 3, 224, 224)))
      .toTensor[Float].split(1)(0))
  }

  "predictImage with more data" should "work properly" in {
    import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
    RNG.setSeed(100)
    val resource = getClass.getClassLoader.getResource("pascal/")
    val ims = (1 to 50).map(x => {
      val im = ImageFeature()
      im(ImageFeature.uri) = x.toString
      im(ImageFeature.imageTensor) = Tensor[Float](3, 24, 24).randn()
      im
    })

    val imageFrame = ImageFrame.array(ims.toArray) -> ImageFrameToSample()
    val model = Sequential()
    model.add(SpatialConvolution(3, 6, 5, 5))
    model.add(Tanh())
    val detection = model.predictImage(imageFrame).toLocal()
    val feature = detection.array.head

    val imageFeatures = detection.array
    val prob = imageFeatures.map(x => x[Tensor[Float]](ImageFeature.predict))
    val data = imageFeatures.map(_[Sample[Float]](ImageFeature.sample))
    prob(0) should be(model.evaluate().forward(data(0).feature.reshape(Array(1, 3, 24, 24)))
      .toTensor[Float].split(1)(0))
    (1 to 20).foreach(x => {
      imageFeatures(x - 1).uri() should be (x.toString)
      if (imageFeatures(x - 1).predict() == null) println(x, imageFeatures(x - 1).predict())
      assert(imageFeatures(x - 1).predict() != null)
    })
  }
}
