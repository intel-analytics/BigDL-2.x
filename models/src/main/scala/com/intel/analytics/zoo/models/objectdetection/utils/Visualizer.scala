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

package com.intel.analytics.zoo.models.objectdetection.utils

import java.io.File
import java.nio.file.Paths

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.ImageFeature
import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat
import com.intel.analytics.bigdl.transform.vision.image.util.BoundingBox
import org.opencv.imgcodecs.Imgcodecs

/**
 * used for image object detection
 * visualize detected bounding boxes and their scores to image
 */
object Visualizer {

  private def visualize(mat: OpenCVMat, rois: Tensor[Float],
    classNames: Array[String], thresh: Float): OpenCVMat = {
    (1 to rois.size(1)).foreach(i => {
      val score = rois.valueAt(i, 2)
      if (score > thresh) {
        val className = classNames(rois.valueAt(i, 1).toInt)
        val bbox = BoundingBox(rois.valueAt(i, 3), rois.valueAt(i, 4),
          rois.valueAt(i, 5), rois.valueAt(i, 6))
        mat.drawBoundingBox(bbox, s"$className $score")
      }
    })
    mat
  }

  private def visualizeDetection(image: Array[Byte],
    uri: String, rois: Tensor[Float], classNames: Array[String],
    thresh: Float = 0.3f, outPath: String = "data/demo"): Unit = {
    require(rois.dim() == 2, "output dim should be 2")
    require(rois.size(2) == 6, "output should have 6 cols, class score xmin ymin xmax ymax")
    val f = new File(outPath)
    if (!f.exists()) {
      f.mkdirs()
    }
    val path = Paths.get(outPath,
      s"detection_${uri.substring(uri.lastIndexOf("/") + 1)}").toString
    val mat = OpenCVMat.fromImageBytes(image)
    visualize(mat, rois, classNames, thresh)
    Imgcodecs.imwrite(path, mat)
    mat.release()
  }

  def draw(imageFeature: ImageFeature, classNames: Array[String],
    thresh: Float = 0.3f, outPath: String = "data/demo"): Unit = {
    val rois = imageFeature.predict().asInstanceOf[Tensor[Float]]
    val uri = imageFeature.uri()
    val image = imageFeature.bytes()
    visualizeDetection(image, uri, rois, classNames, thresh, outPath)
  }
}
