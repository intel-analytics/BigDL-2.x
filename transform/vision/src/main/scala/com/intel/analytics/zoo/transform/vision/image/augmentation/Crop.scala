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

package com.intel.analytics.zoo.transform.vision.image.augmentation

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.zoo.transform.vision.image.opencv.OpenCVMat
import com.intel.analytics.zoo.transform.vision.image.{FeatureTransformer, ImageFeature}
import com.intel.analytics.zoo.transform.vision.util.{BboxUtil, NormalizedBox}
import org.opencv.core.Rect

class Crop(useNormalized: Boolean = true,
           bbox: Option[NormalizedBox] = None, roiKey: Option[String] = None,
           roiGenerator: Option[(ImageFeature => NormalizedBox)] = None)
  extends FeatureTransformer {

  override def transformMat(feature: ImageFeature): Unit = {
    val cropBox = if (bbox.isDefined) {
      bbox.get
    } else if (roiKey.isDefined) {
      var roi = feature(roiKey.get).asInstanceOf[Tensor[Float]]
      if (roi.dim() == 1) {
        roi = BboxUtil.decodeRois(roi)
      }
      if (roi.nElement() > 0) {
        NormalizedBox(roi.valueAt(1, 3), roi.valueAt(1, 4), roi.valueAt(1, 5), roi.valueAt(1, 6))
      } else {
        NormalizedBox(0, 0, 1, 1)
      }
    } else if (roiGenerator.isDefined) {
      roiGenerator.get(feature)
    } else {
      throw new Exception("currently only three mode is accepted, provide the crop bbox," +
        " or crop bbox tensor, or a method that generate crop bbox")
    }
    Crop.transform(feature.opencvMat(), feature.opencvMat(), cropBox, useNormalized)
    if (feature.hasLabel()) {
      feature(ImageFeature.cropBbox) = cropBox
    }
  }
}

object Crop {
  def apply(useNormalized: Boolean = true,
            bbox: Option[NormalizedBox] = None, roiKey: Option[String] = None,
            roiGenerator: Option[(ImageFeature => NormalizedBox)] = None): Crop
  = new Crop(useNormalized, bbox, roiKey, roiGenerator)

  def transform(input: OpenCVMat, output: OpenCVMat, bbox: NormalizedBox,
                useNormalized: Boolean = true): Boolean = {
    val crop = if (useNormalized) {
      val width = input.width
      val height = input.height
      bbox.clipBox(bbox)
      val scaledBox = new NormalizedBox()
      bbox.scaleBox(height, width, scaledBox)
      scaledBox
    } else {
      bbox
    }
    val rect = new Rect(crop.x1.toInt, crop.y1.toInt,
      crop.width().toInt, crop.height().toInt)
    input.submat(rect).copyTo(output)
    true
  }
}

class CenterCrop(cropWidth: Int, cropHeight: Int) extends FeatureTransformer{
  def centerRoi(feature: ImageFeature): NormalizedBox = {
    val mat = feature.opencvMat()
    val height = mat.height().toFloat
    val width = mat.width().toFloat
    val startH = (height - cropHeight) / 2
    val startW = (width - cropWidth) / 2
    NormalizedBox(startW / width, startH / height,
      (startW + cropWidth) / width, (startH + cropHeight) / height)
  }

  val cropper = Crop(roiGenerator = Some(centerRoi))

  override def transformMat(feature: ImageFeature): Unit = {
    cropper.transformMat(feature)
  }
}

object CenterCrop {
  def apply(cropWidth: Int, cropHeight: Int): CenterCrop = new CenterCrop(cropWidth, cropHeight)
}

class RandomCrop(cropWidth: Int, cropHeight: Int) extends FeatureTransformer{
  def randomRoi(feature: ImageFeature): NormalizedBox = {
    val mat = feature.opencvMat()
    val height = mat.height().toFloat
    val width = mat.width().toFloat
    val startH = math.ceil(RNG.uniform(1e-2, height - cropHeight)).toFloat
    val startW = math.ceil(RNG.uniform(1e-2, width - cropWidth)).toFloat
    NormalizedBox(startW / width, startH / height,
      (startW + cropWidth) / width, (startH + cropHeight) / height)
  }

  val cropper = Crop(roiGenerator = Some(randomRoi))

  override def transformMat(feature: ImageFeature): Unit = {
    cropper.transformMat(feature)
  }
}

object RandomCrop {
  def apply(cropWidth: Int, cropHeight: Int): RandomCrop = new RandomCrop(cropWidth, cropHeight)
}
