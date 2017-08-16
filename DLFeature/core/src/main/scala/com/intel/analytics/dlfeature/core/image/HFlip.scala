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

import com.intel.analytics.dlfeature.core.label.LabelTransformMethod
import com.intel.analytics.dlfeature.core.util.MatWrapper
import org.opencv.core.Core

class HFlip(threshProb: Double = 0,
  labelTransformMethod: Option[LabelTransformMethod] = None)
  extends ImageTransformer {

  if (labelTransformMethod.isDefined) setLabelTransfomer(labelTransformMethod.get)

  override def transform(input: MatWrapper, output: MatWrapper, feature: Feature): Boolean = {
    randomOperation(HFlip.transform, input, output, threshProb)
  }
}

object HFlip {
  def apply(threshProb: Double = 0,
    labelTransformMethod: Option[LabelTransformMethod] = None): HFlip
  = new HFlip(threshProb, labelTransformMethod)

  def transform(input: MatWrapper, output: MatWrapper): MatWrapper = {
    Core.flip(input, output, 1)
    output
  }
}
