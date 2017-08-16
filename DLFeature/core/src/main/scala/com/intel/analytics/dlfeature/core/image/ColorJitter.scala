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

package com.intel.analytics.dlfeature.core.image

import com.intel.analytics.dlfeature.core.util.{MatWrapper}
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
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
  randomOrderProb: Double) extends ImageTransformer {

  require(contrastUpper >= contrastLower, "contrast upper must be >= lower.")
  require(contrastLower >= 0, "contrast lower must be non-negative.")
  require(saturationUpper >= saturationLower, "saturation upper must be >= lower.")
  require(saturationLower >= 0, "saturation lower must be non-negative.")

  override def transform(input: MatWrapper, output: MatWrapper, feature: Feature): Boolean = {
    ColorJitter.transform(input, output,
      brightnessProb, brightnessDelta, contrastProb,
      contrastLower, contrastUpper, hueProb, hueDelta, saturationProb,
      saturationLower, saturationUpper, randomOrderProb)
    true
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

  private def randomOperation(operation: ((MatWrapper, MatWrapper, Float) => MatWrapper),
    input: MatWrapper, output: MatWrapper, lower: Double, upper: Double,
    threshProb: Double): Unit = {
    val prob = RNG.uniform(0, 1)
    if (prob < threshProb) {
      val delta = RNG.uniform(lower, upper).toFloat
      operation(input, output, delta)
    } else {
      if (input != output) input.copyTo(output)
    }
  }

  private def randomOperation(operation: ((MatWrapper, MatWrapper) => MatWrapper),
    input: MatWrapper, output: MatWrapper,
    threshProb: Double): Unit = {
    val prob = RNG.uniform(0, 1)
    if (prob < threshProb) {
      operation(input, output)
    } else {
      if (input != output) input.copyTo(output)
    }
  }


  def transform(input: MatWrapper, output: MatWrapper,
    brightnessProb: Double, brightnessDelta: Double,
    contrastProb: Double, contrastLower: Double, contrastUpper: Double,
    hueProb: Double, hueDelta: Double,
    saturationProb: Double, saturationLower: Double, saturationUpper: Double,
    randomOrderProb: Double): MatWrapper = {
    val prob = RNG.uniform(0, 1)
    if (prob > 0.5) {
      // Do random brightness distortion.
      randomOperation(Brightness.transform, input, output,
        -brightnessDelta, brightnessDelta, brightnessProb)
      // Do random contrast distortion.
      randomOperation(Contrast.transform, output, output,
        contrastLower, contrastUpper, contrastProb)
      // Do random saturation distortion.
      randomOperation(Saturation.transform, output, output,
        saturationLower, saturationUpper, contrastProb)
      // Do random hue distortion.
      randomOperation(Hue.transform, output, output, -hueDelta, hueDelta, hueProb)
      // Do random reordering of the channels.
      randomOperation(ChannelOrder.transform, output, output, randomOrderProb)
    } else {
      // Do random brightness distortion.
      randomOperation(Brightness.transform, input, output,
        -brightnessDelta, brightnessDelta, brightnessProb)
      // Do random saturation distortion.
      randomOperation(Saturation.transform, output, output,
        saturationLower, saturationUpper, contrastProb)
      // Do random hue distortion.
      randomOperation(Hue.transform, output, output, -hueDelta, hueDelta, hueProb)
      // Do random contrast distortion.
      randomOperation(Contrast.transform, output, output,
        contrastLower, contrastUpper, contrastProb)
      // Do random reordering of the channels.
      randomOperation(ChannelOrder.transform, output, output, randomOrderProb)
    }
    output
  }
}
