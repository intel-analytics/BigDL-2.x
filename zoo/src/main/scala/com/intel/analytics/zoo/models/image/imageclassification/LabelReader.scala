/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.models.image.imageclassification

import com.intel.analytics.zoo.models.image.imageclassification.Dataset.Imagenet
import com.intel.analytics.zoo.models.common.ModelLabelReader


object LabelReader extends ModelLabelReader {
  def readImagenetlLabelMap(modelName: String = null): Map[Int, String] = {
    modelName match {
      case "inception-v3" => readLabelMap("/imagenet_2015_classname.txt")
      case _ => readLabelMap("/imagenet_classname.txt")
    }
  }

  def apply(dataset: String, modelName: String = null): Map[Int, String] = {
    Dataset(dataset) match {
      case Imagenet => readImagenetlLabelMap(modelName)
      case _ =>
        throw new Exception("currently only support Imagenet dataset in" +
          " Analytics zoo Image classification models")
    }
  }
}
