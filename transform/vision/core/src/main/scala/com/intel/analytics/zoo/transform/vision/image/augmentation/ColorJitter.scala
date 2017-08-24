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

import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import com.intel.analytics.zoo.transform.vision.image._
/**
 * Random adjust brightness, contrast, hue, saturation
 * @param brightnessProb probability to adjust brightness
 * @param brightnessDelta brightness parameter
 * @param contrastProb probability to adjust contrast
 * @param contrastLower contrast lower parameter
 * @param contrastUpper contrast upper parameter
 * @param hueProb probability to adjust hue
 * @param hueDelta hue parameter
 * @param saturationProb probability to adjust saturation
 * @param saturationLower saturation lower parameter
 * @param saturationUpper saturation upper parameter
 * @param randomOrderProb random order for different operation
 */
class ColorJitter(
  brightnessProb: Double, brightnessDelta: Double,
  contrastProb: Double, contrastLower: Double, contrastUpper: Double,
  hueProb: Double, hueDelta: Double,
  saturationProb: Double, saturationLower: Double, saturationUpper: Double,
  randomOrderProb: Double) extends FeatureTransformer {

  require(contrastUpper >= contrastLower, "contrast upper must be >= lower.")
  require(contrastLower >= 0, "contrast lower must be non-negative.")
  require(saturationUpper >= saturationLower, "saturation upper must be >= lower.")
  require(saturationLower >= 0, "saturation lower must be non-negative.")

  private val brightness = RandomTransformer(Brightness(-brightnessDelta, brightnessDelta), brightnessProb)
  private val contrast = RandomTransformer(Contrast(contrastLower, contrastUpper), contrastProb)
  private val saturation = RandomTransformer(Saturation(saturationLower, saturationUpper), saturationProb)
  private val hue = RandomTransformer(Hue(-hueDelta, hueDelta), hueProb)
  private val channelOrder = RandomTransformer(ChannelOrder(), randomOrderProb)

  private val order1 = brightness -> contrast -> saturation -> hue -> channelOrder
  private val order2 = brightness -> saturation -> hue -> contrast -> channelOrder

  override def transform(feature: ImageFeature): Unit = {
    val prob = RNG.uniform(0, 1)
    if (prob > 0.5) {
      order1(feature)
    } else {
      order2(feature)
    }
  }
}

object ColorJitter {
  def apply(
    brightnessProb: Double, brightnessDelta: Double,
    contrastProb: Double, contrastLower: Double, contrastUpper: Double,
    hueProb: Double, hueDelta: Double,
    saturationProb: Double, saturationLower: Double, saturationUpper: Double,
    randomOrderProb: Double
  ): ColorJitter =
    new ColorJitter(brightnessProb, brightnessDelta, contrastProb,
      contrastLower, contrastUpper, hueProb, hueDelta, saturationProb,
      saturationLower, saturationUpper, randomOrderProb)

  def apply(): ColorJitter =
    ColorJitter(0.5, 32, 0.5, 0.5, 1.5, 0.5, 18, 0.5, 0.5, 1.5, 0)
}
