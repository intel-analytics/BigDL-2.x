package com.intel.analytics.zoo.feature.image3d

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.ImageFeature
import com.intel.analytics.bigdl.utils.RandomGenerator._
import org.scalatest.{FlatSpec, Matchers}

/**
  * Created by jwang on 6/25/18.
  */
class Crop3DSpec extends FlatSpec with Matchers{
  "A CropTransformer" should "generate correct output." in{
    val seed = 100
    RNG.setSeed(seed)
    val input = Tensor[Float](60, 70, 80)
    input.apply1(e => RNG.uniform(0, 1).toFloat)
    val image = ImageFeature3D(input)
    val start = Array[Int](10, 20, 20)
    val patchSize = Array[Int](21, 31, 41)
    val cropper = Crop3D(start, patchSize)
    val output = cropper.transform(image)
    val result = input.narrow(1, 10, 21).narrow(2, 20, 31).narrow(3, 20, 41)
      .clone().storage().array()
    output[Tensor[Float]](ImageFeature.imageTensor).storage().array() should be(result)
  }

  "A RandomCropTransformer" should "generate correct output." in{
    val seed = 100
    RNG.setSeed(seed)
    val input = Tensor[Float](60, 70, 80)
    input.apply1(e => RNG.uniform(0, 1).toFloat)
    val image = ImageFeature3D(input)
    val cropper = RandomCrop3D(20, 30, 40)
    val output = cropper.transform(image)
    assert(output[Tensor[Float]](ImageFeature.imageTensor).storage().array().length == 20 *30*40)
  }
}
