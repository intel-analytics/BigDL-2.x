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

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{DoubleType, FloatType, Tensor}
import scala.reflect.ClassTag
import com.intel.analytics.zoo.transform.vision.image3d._

/*
 * Affine transformer implements affine transformation on a given tensor.
 * To avoid defects in resampling, the mapping is from destination to source.
 * dst(z,y,x) = src(f(z),f(y),f(x)) where f: dst -> src
 *
 * @param mat [Tensor[Double], dim: DxHxW] defines affine transformation from dst to src.
 * @param mode [String, "bilinear"]resampling method. Currently only bilinear available.
 * @param translation [Tensor[Double], dim: 3, default: (0,0,0)] defines translation in each axis.
 * @param clamp_mode [String, (default: "clamp",'padding')] defines how to handle interpolation off the input image.
 * @param pad_val [Double, default: 0] defines padding value when clamp_mode="padding". Setting this value when clamp_mode="clamp" will cause an error.
 */

object AffineTransform{
  def apply(mat: Tensor[Double],
            translation: Tensor[Double] = Tensor[Double](3).fill(0),
            clamp_mode: String = "clamp",
            pad_val: Double = 0): AffineTransform =
      new AffineTransform(mat, translation, clamp_mode, pad_val)
}

class AffineTransform(mat: Tensor[Double],
                        translation: Tensor[Double],
                        clamp_mode: String,
                        pad_val: Double)
extends FeatureTransformer {
  private val _clamp_mode = clamp_mode match {
    case "clamp" => 1
    case "padding" => 2
  }

  override def transformTensor(tensor: Tensor[Float]): Tensor[Float] = {
//  def apply[@specialized(Float, Double)T: ClassTag](src: Tensor[T], dims: Array[Int])
//  (implicit ev: TensorNumeric[T]): Tensor[T] = {
//    require(dims.size==3,"Invalid dimension")
    val dst = Tensor[Float](tensor.size())
    val depth = dst.size(1)
    val height = dst.size(2)
    val width = dst.size(3)
    var grid_xyz = Tensor[Double](Array[Int](3, depth, height, width))
    val cz = (depth + 1)/2.0
    val cy = (height + 1)/2.0
    val cx = (width + 1)/2.0
    for(z <- 1 to depth; y <- 1 to height; x <- 1 to width) {
      grid_xyz.setValue(1, z, y, x, cz-z)
      grid_xyz.setValue(2, z, y, x, cy-y)
      grid_xyz.setValue(3, z, y, x, cx-x)
    }
    val view_xyz = grid_xyz.reshape(Array[Int](3, depth * height * width))
    val field = mat * view_xyz
    grid_xyz = grid_xyz.sub(field.reshape(Array[Int](3, depth, height, width)))
    val translation_mat = Tensor[Double](Array[Int](3, depth, height, width))
    translation_mat(1).fill(translation.valueAt(1))
    translation_mat(2).fill(translation.valueAt(2))
    translation_mat(3).fill(translation.valueAt(3))
    grid_xyz(1) = grid_xyz(1).sub(translation_mat(1))
    grid_xyz(2) = grid_xyz(2).sub(translation_mat(2))
    grid_xyz(3) = grid_xyz(3).sub(translation_mat(3))
    val offset_mode = true
    val warp_transformer = WarpTransformer(grid_xyz, offset_mode, clamp_mode, pad_val)
    warp_transformer(tensor, dst)
    dst
  }// end apply
}//end class
