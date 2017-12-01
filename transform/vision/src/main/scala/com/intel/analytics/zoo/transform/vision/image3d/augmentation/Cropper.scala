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

import com.intel.analytics.zoo.transform.vision.image3d._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag
import java.util.Calendar

import com.intel.analytics.bigdl.utils.RandomGenerator._
/**
 * Crop a patch from an tensor from 'start' of patch size. The patch size should be less than
 * the image size.
 * @param start
 * @param patchSize
 */
object Crop {
  def apply(start: Array[Int], patchSize: Array[Int]): Crop =
    new Crop(start, patchSize)

  def crop(tensor: Tensor[Float], start: Array[Int], patchSize: Array[Int]): Tensor[Float] = {
    require(start(0) <= tensor.size(1) && start(1) <= tensor.size(2) && start(2) <= tensor.size(3),
      "Cropping indices out of bounds.")
    require(start(0) + patchSize(0) - 1  <= tensor.size(1)
      && start(1) + patchSize(1) - 1 <= tensor.size(2)
      && start(2) + patchSize(2) - 1 <= tensor.size(3), "Cropping indices out of bounds.")
    tensor.narrow(1, start(0), patchSize(0))
      .narrow(2, start(1), patchSize(1))
      .narrow(3, start(2), patchSize(2))
  }
}

class Crop(start: Array[Int], patchSize: Array[Int])
  extends FeatureTransformer{
  require(start.size == 3 && patchSize.size == 3,
    "'start' array and 'patchSize' array should have dim 3.")
  require(patchSize(0) >= 0 && patchSize(1) >= 0 && patchSize(2) >= 0,
    "'patchSize' values should be nonnegative.")

  require(start.map(t => t >= 0).reduce((a, b) => a && b),
    "'start' values should be nonnegative.")

  override def transformTensor(tensor: Tensor[Float]): Tensor[Float] = {
    Crop.crop(tensor, start, patchSize)
  }
}

object RandomCrop {
  def apply(cropDepth: Int, cropHeight: Int, cropWidth: Int): RandomCrop =
    new RandomCrop(cropDepth, cropHeight, cropWidth)
}

class RandomCrop(cropDepth: Int, cropHeight: Int, cropWidth: Int)
  extends FeatureTransformer{

  override def transformTensor(tensor: Tensor[Float]): Tensor[Float] = {
    require(tensor.dim == 3,
      "the transformed image array should have dim 3.")
    require(tensor.size(1) >= cropDepth,
      "the transformed image depth should be larger than cropped depth.")
    require(tensor.size(2) >= cropHeight,
      "the transformed image width should be larger than cropped width.")
    require(tensor.size(3) >= cropWidth,
      "the transformed image height should be larger than cropped height.")
    val startD = math.ceil(RNG.uniform(1e-2, tensor.size(1) - cropDepth)).toInt
    val startH = math.ceil(RNG.uniform(1e-2, tensor.size(2) - cropHeight)).toInt
    val startW = math.ceil(RNG.uniform(1e-2, tensor.size(3) - cropWidth)).toInt
    Crop.crop(tensor,
      Array[Int](startD, startH, startW),
      Array[Int](cropDepth, cropHeight, cropWidth))
  }
}

object CenterCrop {
  def apply(cropDepth: Int, cropHeight: Int, cropWidth: Int): CenterCrop =
    new CenterCrop(cropDepth, cropHeight, cropWidth)
}

class CenterCrop(cropDepth: Int, cropHeight: Int, cropWidth: Int)
  extends FeatureTransformer{

  override def transformTensor(tensor: Tensor[Float]): Tensor[Float] = {
    require(tensor.dim == 3,
      "the transformed image array should have dim 3.")
    require(tensor.size(1) >= cropDepth,
      "the transformed image depth should be larger than cropped depth.")
    require(tensor.size(2) >= cropHeight,
      "the transformed image width should be larger than cropped width.")
    require(tensor.size(3) >= cropWidth,
      "the transformed image height should be larger than cropped height.")
    val startD = (tensor.size(1) - cropDepth)/2
    val startH = (tensor.size(2) - cropHeight)/2
    val startW = (tensor.size(3) - cropWidth)/2
    Crop.crop(tensor,
      Array[Int](startD, startH, startW),
      Array[Int](cropDepth, cropHeight, cropWidth))
  }
}