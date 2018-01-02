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

package com.intel.analytics.zoo.models.imageclassification.util

import com.intel.analytics.zoo.models.imageclassification.util.Dataset.Imagenet
import com.intel.analytics.zoo.models.util.ModelLabelReader


object LabelReader extends ModelLabelReader {
  def readImagenetlLabelMap(): Map[Int, String] = {
    readLabelMap("/imagenet_classname.txt")
  }

  def apply(dataset: String): Map[Int, String] = {
    Dataset(dataset) match {
      case Imagenet =>
        readImagenetlLabelMap()
      case _ =>
        throw new Exception("currently only support Imagenet dataset in" +
          " BigDL Image classification models")
    }
  }
}
