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
package com.intel.analytics.bigdl.dllib.feature.common

import com.intel.analytics.bigdl.dllib.feature.dataset.Transformer
import com.intel.analytics.bigdl.dllib.feature.transform.vision.image.ImageFeature
import com.intel.analytics.bigdl.dllib.feature.image.ImageSet
import com.intel.analytics.bigdl.dllib.feature.text.{TextFeature, TextSet}
import com.intel.analytics.bigdl.dllib.utils.Log4Error
import org.apache.commons.lang3.SerializationUtils

/**
 * [[Preprocessing]] defines data transform action during feature preprocessing.
 * Multiple [[Preprocessing]] can be combined into a [[ChainedPreprocessing]].
 * E.g., FeatureStep1[A, B] -> FeatureStep2[B, C] yield a ChainedFeatureSteps[A, C]
 *
 * @tparam A input data type
 * @tparam B output data type
 */
trait Preprocessing[A, B] extends Transformer[A, B] {
  // scalastyle:off methodName
  // scalastyle:off noSpaceBeforeLeftBracket
  def -> [C](other: Preprocessing[B, C]): Preprocessing[A, C] = {
    new ChainedPreprocessing(this, other)
  }
  // scalastyle:on noSpaceBeforeLeftBracket
  // scalastyle:on methodName

  def clonePreprocessing(): Preprocessing[A, B] = {
    SerializationUtils.clone(this)
  }

//  def apply(imageSet: ImageSet): ImageSet = {
//    if (this.isInstanceOf[Preprocessing[ImageFeature, ImageFeature]]) {
//      imageSet.transform(this.asInstanceOf[Preprocessing[ImageFeature, ImageFeature]])
//    } else {
//      Log4Error.invalidOperationError(false,"We expect " +
//        "Preprocessing[ImageFeature, ImageFeature] here")
//    }
//  }

def apply(imageSet: ImageSet): ImageSet = {
    this match {
      case xs: com.intel.analytics.bigdl.dllib.feature.common.Preprocessing[com.intel.analytics.
      bigdl.dllib.feature.transform.vision.image.ImageFeature, com.intel.analytics.bigdl.dllib.
      feature.transform.vision.image.ImageFeature] =>
        imageSet.transform(this.asInstanceOf[Preprocessing[ImageFeature, ImageFeature]])
      case _ =>
        Log4Error.unKnowExceptionError(false,
          "We expect Preprocessing[ImageFeature, ImageFeature] here")
        null
}
//    if (this.isInstanceOf[Preprocessing[ImageFeature, ImageFeature]]) {
//      imageSet.transform(this.asInstanceOf[Preprocessing[ImageFeature, ImageFeature]])
//    } else {
//      Log4Error.invalidOperationError(false,"We expect " +
//        "Preprocessing[ImageFeature, ImageFeature] here")
//    }
  }


  def apply(textSet: TextSet): TextSet = {
    this match {
      case textTransformer: Preprocessing[TextFeature, TextFeature] =>
        textSet.transform(textTransformer)
      case _ =>
        Log4Error.unKnowExceptionError(false,
          "We expect Preprocessing[TextFeature, TextFeature] here")
        null
    }
  }
}

/**
 * chains two Preprocessing together. The output type of the first
 * Preprocessing should be the same with the input type of the second Preprocessing.
 *
 * @param first first Preprocessing
 * @param last last Preprocessing
 * @tparam A input type of the first Preprocessing
 * @tparam B output type of the first Preprocessing, as well as the input type of the last
 *           Preprocessing
 * @tparam C output of the last Preprocessing
 */
class ChainedPreprocessing[A, B, C](first: Preprocessing[A, B], last: Preprocessing[B, C])
  extends Preprocessing[A, C] {
  override def apply(prev: Iterator[A]): Iterator[C] = {
    last(first(prev))
  }
}


