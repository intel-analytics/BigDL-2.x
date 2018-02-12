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

package com.intel.analytics.zoo.pipeline.utils

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.{FeatureTransformer, ImageFeature}
import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat
import com.intel.analytics.bigdl.transform.vision.image.util.BoundingBox
import com.intel.analytics.bigdl.utils.RandomGenerator
import org.opencv.core.{Core, Point, Scalar}
import org.opencv.imgproc.Imgproc
import Visualizer._

class Visualizer(labelMap: Map[Int, String], thresh: Float = 0.3f,
  encoding: String = "png", outKey: String = Visualizer.visualized) extends FeatureTransformer {
  override def transformMat(imageFeature: ImageFeature): Unit = {
    val image = imageFeature.bytes()
    val out = imageFeature[(Tensor[Float], Tensor[Float],
      Tensor[Float], Array[Tensor[Float]])]("unmode")
    val mat = OpenCVMat.fromImageBytes(image)
    Visualizer.visualize(mat, out._1, out._3, out._2, labelMap, thresh)
    val imageWithMask = drawMask(mat, out._4, out._3, thresh)
    imageFeature(outKey) = OpenCVMat.imencode(imageWithMask, encoding)
  }

}

object Visualizer {

  val visualized = "visualized"

  def apply(labelMap: Map[Int, String], thresh: Float = 0.3f,
    encoding: String = "png", outKey: String = visualized): Visualizer =
    new Visualizer(labelMap, thresh, encoding, outKey)


  def visualize(mat: OpenCVMat, rois: Tensor[Float],
    labelMap: Map[Int, String], thresh: Double): OpenCVMat = {
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

  def visualize(mat: OpenCVMat,
    rois: Tensor[Float], scores: Tensor[Float], classIds: Tensor[Float],
    labelMap: Map[Int, String], thresh: Double): OpenCVMat = {
    (1 to rois.size(1)).foreach(i => {
      val score = scores.valueAt(i)
      if (score > thresh) {
        val className = labelMap(classIds.valueAt(i).toInt)
        val bbox = BoundingBox(rois.valueAt(i, 1), rois.valueAt(i, 2),
          rois.valueAt(i, 3), rois.valueAt(i, 4))
        drawBoundingBox(mat, bbox, s"$className $score")
      }
    })
    mat
  }

  def drawBoundingBox(image: OpenCVMat,
    bbox: BoundingBox, text: String,
    font: Int = Core.FONT_HERSHEY_COMPLEX_SMALL,
    boxColor: (Double, Double, Double) = (0, 255, 0),
    textColor: (Double, Double, Double) = (255, 255, 255),
    opacity: Double = 1): this.type = {
    var imageCopy: OpenCVMat = null
    if (opacity != 1) {
      imageCopy = new OpenCVMat()
      image.copyTo(imageCopy)
    }
    Imgproc.rectangle(image,
      new Point(bbox.x1, bbox.y1),
      new Point(bbox.x2, bbox.y2),
      new Scalar(boxColor._1, boxColor._2, boxColor._3), 1)
    Imgproc.putText(image, text,
      new Point(bbox.x1, bbox.y1 - 2),
      font, 0.8,
      new Scalar(textColor._1, textColor._2, textColor._3), 1)
    if (opacity != 1) {
      Core.addWeighted(image, opacity, imageCopy, 1 - opacity, 0, image)
      imageCopy.release()
    }
    this
  }

  def drawMask(image: OpenCVMat, masks: Array[Tensor[Float]], scores: Tensor[Float] = null,
    thresh: Double = 0.3,
    opacity: Float = 0.5f): OpenCVMat = {
    val images = OpenCVUtil.toTensor(image)
    var i = 1
    masks.foreach(mask => {
      if (scores == null || scores.valueAt(i) > thresh) drawSingleMask(opacity, images, mask)
      i += 1
    })

    val mat = OpenCVMat.fromTensor(images)
    mat.copyTo(image)
    image
  }

  def drawSingleMask(opacity: Float, images: Tensor[Float], mask: Tensor[Float]): Unit = {
    val r = RandomGenerator.RNG.uniform(0, 255)
    val g = RandomGenerator.RNG.uniform(0, 255)
    val b = RandomGenerator.RNG.uniform(0, 255)
    require(mask.dim() == 2, s"there should be two dim in mask, while got ${mask.dim()}")

    (1 to images.size(1)).foreach(h => {
      (1 to images.size(2)).foreach(w => {
        if (mask.valueAt(h, w) == 1) {
          images.setValue(h, w, 1,
            images.valueAt(h, w, 1) * opacity + (1 - opacity) * b.toFloat)
          images.setValue(h, w, 2,
            images.valueAt(h, w, 2) * opacity + (1 - opacity) * g.toFloat)
          images.setValue(h, w, 3,
            images.valueAt(h, w, 3) * opacity + (1 - opacity) * r.toFloat)
        }
      })
    })
  }
}


