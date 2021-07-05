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

package com.intel.analytics.zoo.models.image.objectdetection

import com.intel.analytics.zoo.models.image.objectdetection.ObjectDetectorDataset.{Coco, Pascal}
import com.intel.analytics.zoo.models.common.ModelLabelReader

private[zoo] object LabelReader extends ModelLabelReader {

  /**
   * load pascal label map
   */
  def readPascalLabelMap(): Map[Int, String] = {
    readLabelMap("/pascal_classname.txt")
  }

  /**
   * load coco label map
   */
  def readCocoLabelMap(): Map[Int, String] = {
    readLabelMap("/coco_classname.txt")
  }

  def apply(dataset: String): Map[Int, String] = {
    ObjectDetectorDataset(dataset) match {
      case Pascal =>
        readPascalLabelMap()
      case Coco =>
        readCocoLabelMap()
      case _ =>
        throw new Exception("currently only support Pascal and Coco dataset in" +
          " Analytics zoo object detection models")
    }
  }
}
