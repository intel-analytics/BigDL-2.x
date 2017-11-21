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

package com.intel.analytics.zoo.pipeline.common.dataset.roiimage

import java.awt.image.BufferedImage
import java.awt.{BasicStroke, Color, Font, Graphics2D}
import java.io.File
import java.nio.file.Paths
import javax.imageio.ImageIO

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.transform.vision.image.ImageFeature
import com.intel.analytics.zoo.transform.vision.image.opencv.OpenCVMat
import org.opencv.core.{Core, Point, Scalar}
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc

/**
 * used for image object detection
 * visualize detected bounding boxes and their scores to image
 */
object Visualizer {
  private val bgColor = new Color(0, 0, 1, 127)
  private val font = new Font("Helvetica", Font.PLAIN, 14)
  private val stroke = new BasicStroke(3)

  private def vis(imgPath: String, output: Tensor[Float],
    savePath: String, classNames: Array[String], thresh: Float = 0.3f): Unit = {
    var img: BufferedImage = null
    var g2d: Graphics2D = null

    def loadImage() = {
      img = ImageIO.read(new File(imgPath))
      g2d = img.createGraphics
      g2d.setFont(font)
      g2d.setStroke(stroke)
    }

    var i = 1
    while (i <= output.size(1)) {
      val score = output.valueAt(i, 2)
      if (score > thresh) {
        if (g2d == null) {
          loadImage()
        }
        val className = classNames(output.valueAt(i, 1).toInt)
        draw(g2d, output.valueAt(i, 3).toInt, output.valueAt(i, 4).toInt,
          output.valueAt(i, 5).toInt - output.valueAt(i, 3).toInt,
          output.valueAt(i, 6).toInt - output.valueAt(i, 4).toInt,
          s"$className ${ "%.3f".format(score) }")
      }
      i += 1
    }
    if (g2d != null) {
      ImageIO.write(img, savePath.substring(savePath.lastIndexOf(".") + 1), new File(savePath))
      g2d.dispose()
    }
  }

  private def draw(img: Graphics2D, x1: Int, y1: Int,
    width: Int, height: Int, title: String): Unit = {
    img.setColor(Color.RED)
    img.drawRect(x1, y1, width, height)

    val fm = img.getFontMetrics()
    val rect = fm.getStringBounds(title, img)

    img.setColor(bgColor)
    img.fillRect(x1, y1 - 2 - fm.getAscent,
      rect.getWidth.toInt,
      rect.getHeight.toInt)
    img.setColor(Color.WHITE)
    img.drawString(title, x1, y1 - 2)
  }

  @deprecated
  def visDetection(imagePath: String, output: Tensor[Float], classNames: Array[String],
    thresh: Float = 0.3f, outPath: String = "data/demo"): Unit = {
    require(output.dim() == 2, "output dim should be 2")
    require(output.size(2) == 6, "output should have 6 cols, class score xmin ymin xmax ymax")
    val f = new File(outPath)
    if (!f.exists()) {
      f.mkdirs()
    }
    val path = Paths.get(outPath,
      s"detection_${ imagePath.substring(imagePath.lastIndexOf("/") + 1) }").toString
    vis(imagePath, output, path, classNames, thresh)
  }

  private def visualize(mat: OpenCVMat, rois: Tensor[Float],
    classNames: Array[String], thresh: Float): OpenCVMat = {
    val font = Core.FONT_HERSHEY_COMPLEX_SMALL
    var i = 1
    while (i <= rois.size(1)) {
      val score = rois.valueAt(i, 2)
      if (score > thresh) {
        val className = classNames(rois.valueAt(i, 1).toInt)
        Imgproc.rectangle(mat, new Point(rois.valueAt(i, 3).toInt,
          rois.valueAt(i, 4).toInt),
          new Point(rois.valueAt(i, 5).toInt,
            rois.valueAt(i, 6).toInt),
          new Scalar(0, 255, 0), 3)
        Imgproc.putText(mat, s"$className $score", new Point(rois.valueAt(i, 3).toInt,
          rois.valueAt(i, 4).toInt - 2), font, 1, new Scalar(255, 255, 255), 1)
      }
      i += 1
    }
    mat
  }

  def visualizeDetection(image: Array[Byte],
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
    val mat = OpenCVMat.toMat(image)
    visualize(mat, rois, classNames, thresh)
    Imgcodecs.imwrite(path, mat)
    mat.release()
  }

  def visualizeImageFeature(imageFeature: ImageFeature, classNames: Array[String],
    thresh: Float = 0.3f, outPath: String = "data/demo"): Unit = {
    val rois = imageFeature.tensorFeature()
    val uri = imageFeature.uri()
    val image = imageFeature.bytes()
    visualizeDetection(image, uri, rois, classNames, thresh, outPath)
  }
}

