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

package com.intel.analytics.zoo.feature.core.image

import java.util.Random

import com.intel.analytics.zoo.feature.core.util.MatWrapper
import org.apache.log4j.Logger
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

/**
 *
 * @param resizeH
 * @param resizeW
 * @param resizeMode if resizeMode = -1, random select a mode from
 * (Imgproc.INTER_LINEAR, Imgproc.INTER_CUBIC, Imgproc.INTER_AREA,
 *                   Imgproc.INTER_NEAREST, Imgproc.INTER_LANCZOS4)
 */
class Resize(resizeH: Int, resizeW: Int,
  resizeMode: Int = Imgproc.INTER_LINEAR)
  extends FeatureTransformer {

  private val interpMethods = Array(Imgproc.INTER_LINEAR, Imgproc.INTER_CUBIC, Imgproc.INTER_AREA,
    Imgproc.INTER_NEAREST, Imgproc.INTER_LANCZOS4)

  override def transform(feature: Feature): Unit = {
    val interpMethod = if (resizeMode == -1) {
      interpMethods(new Random().nextInt(interpMethods.length))
    } else {
      resizeMode
    }
    try {
      Resize.transform(feature.inputMat(), feature.inputMat(), resizeW, resizeH, interpMethod)
    } catch {
      case e: Exception =>
        Resize.logger.error(e)
        feature.inputMat().create(resizeH, resizeW, feature.inputMat().`type`())
    }
  }
}

object Resize {
  val logger = Logger.getLogger(getClass)

  def apply(resizeH: Int, resizeW: Int,
    resizeMode: Int = Imgproc.INTER_LINEAR): Resize =
    new Resize(resizeH, resizeW, resizeMode)

  def transform(input: MatWrapper, output: MatWrapper, resizeW: Int, resizeH: Int,
    mode: Int = Imgproc.INTER_LINEAR)
  : MatWrapper = {
    Imgproc.resize(input, output, new Size(resizeW, resizeH), 0, 0, mode)
    output
  }
}

