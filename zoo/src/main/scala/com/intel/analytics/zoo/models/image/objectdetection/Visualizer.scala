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

package com.intel.analytics.zoo.models.objectdetection

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.{FeatureTransformer, ImageFeature}
import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat
import com.intel.analytics.bigdl.transform.vision.image.util.BoundingBox
import com.intel.analytics.zoo.feature.image.ImageSet

/**
 * used for image object detection
 * visualize detected bounding boxes and their scores to image
 */
class Visualizer(labelMap: Map[Int, String], thresh: Float = 0.3f,
  encoding: String = "png", outKey: String = Visualizer.visualized) extends FeatureTransformer {
  override def transformMat(imageFeature: ImageFeature): Unit = {
    val rois = imageFeature.predict().asInstanceOf[Tensor[Float]]
    val uri = imageFeature.uri()
    val image = imageFeature.bytes()
    val imageFile = visualizeDetection(image, uri, rois)
    imageFeature(outKey) = imageFile
  }


  private def visualize(mat: OpenCVMat, rois: Tensor[Float]): OpenCVMat = {
    (1 to rois.size(1)).foreach(i => {
      val score = rois.valueAt(i, 2)
      if (score > thresh) {
        val className = labelMap(rois.valueAt(i, 1).toInt)
        val bbox = BoundingBox(rois.valueAt(i, 3), rois.valueAt(i, 4),
          rois.valueAt(i, 5), rois.valueAt(i, 6))
        mat.drawBoundingBox(bbox, s"$className $score")
      }
    })
    mat
  }

  private def visualizeDetection(image: Array[Byte],
    uri: String, rois: Tensor[Float]): Array[Byte] = {
    if (rois.dim() != 2) return image
    require(rois.dim() == 2, "output dim should be 2")
    require(rois.size(2) == 6, "output should have 6 cols, class score xmin ymin xmax ymax")
    var mat: OpenCVMat = null
    try {
      mat = OpenCVMat.fromImageBytes(image)
      visualize(mat, rois)
      OpenCVMat.imencode(mat, encoding)
    } finally {
      if (mat != null) mat.release()
    }
  }

  def apply(imageSet: ImageSet): ImageSet = {
    imageSet.transform(this)
  }
}


object Visualizer {

  val visualized = "visualized"

  def apply(labelMap: Map[Int, String], thresh: Float = 0.3f,
    encoding: String = "png", outKey: String = visualized): Visualizer =
    new Visualizer(labelMap, thresh, encoding, outKey)
}
