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

package com.intel.analytics.zoo.transform.vision.label.roi

import com.intel.analytics.zoo.transform.vision.image.{FeatureTransformer, ImageFeature}
import com.intel.analytics.zoo.transform.vision.util.{BboxUtil, NormalizedBox}

import scala.collection.mutable.ArrayBuffer


case class RoiNormalize() extends FeatureTransformer {
  override def transformMat(feature: ImageFeature): Unit = {
    val height = feature.getHeight()
    val width = feature.getWidth()
    val label = feature(ImageFeature.label).asInstanceOf[RoiLabel]
    BboxUtil.scaleBBox(label.bboxes, 1.0f / height, 1.0f / width)
  }
}


case class RoiCrop() extends FeatureTransformer {
  override def transformMat(feature: ImageFeature): Unit = {
    val height = feature.getHeight()
    val width = feature.getWidth()
    val bbox = feature(ImageFeature.cropBbox).asInstanceOf[NormalizedBox]
    val target = feature(ImageFeature.label).asInstanceOf[RoiLabel]
    val transformedAnnot = new ArrayBuffer[NormalizedBox]()
    // Transform the annotation according to crop_bbox.
    AnnotationTransformer.transformAnnotation(width, height, bbox, false, target,
      transformedAnnot)

    target.bboxes.resize(transformedAnnot.length, 4)
    target.classes.resize(2, transformedAnnot.length)

    var i = 1
    while (i <= transformedAnnot.length) {
      target.bboxes.setValue(i, 1, transformedAnnot(i - 1).x1)
      target.bboxes.setValue(i, 2, transformedAnnot(i - 1).y1)
      target.bboxes.setValue(i, 3, transformedAnnot(i - 1).x2)
      target.bboxes.setValue(i, 4, transformedAnnot(i - 1).y2)
      target.classes.setValue(1, i, transformedAnnot(i - 1).label)
      target.classes.setValue(2, i, transformedAnnot(i - 1).difficult)
      i += 1
    }
  }
}

case class RoiHFlip() extends FeatureTransformer {
  override def transformMat(feature: ImageFeature): Unit = {
    require(feature.hasLabel())
    val roiLabel = feature.getLabel[RoiLabel]
    var i = 1
    while (roiLabel.bboxes.nElement() > 0 && i <= roiLabel.bboxes.size(1)) {
      val x1 = 1 - roiLabel.bboxes.valueAt(i, 1)
      roiLabel.bboxes.setValue(i, 1, 1 - roiLabel.bboxes.valueAt(i, 3))
      roiLabel.bboxes.setValue(i, 3, x1)
      i += 1
    }
  }
}

case class RoiExpand() extends FeatureTransformer {

  override def transformMat(feature: ImageFeature): Unit = {
    require(feature.hasLabel())
    val transformedAnnot = new ArrayBuffer[NormalizedBox]()
    val expandBbox = feature(ImageFeature.expandBbox).asInstanceOf[NormalizedBox]
    val height = feature.getHeight()
    val width = feature.getWidth()
    val target = feature.getLabel[RoiLabel]
    AnnotationTransformer.transformAnnotation(width, height, expandBbox, false,
      target, transformedAnnot)


    target.bboxes.resize(transformedAnnot.length, 4)
    target.classes.resize(2, transformedAnnot.length)

    var i = 1
    while (i <= transformedAnnot.length) {
      target.bboxes.setValue(i, 1, transformedAnnot(i - 1).x1)
      target.bboxes.setValue(i, 2, transformedAnnot(i - 1).y1)
      target.bboxes.setValue(i, 3, transformedAnnot(i - 1).x2)
      target.bboxes.setValue(i, 4, transformedAnnot(i - 1).y2)
      target.classes.setValue(1, i, transformedAnnot(i - 1).label)
      target.classes.setValue(2, i, transformedAnnot(i - 1).difficult)
      i += 1
    }
  }
}

object AnnotationTransformer {
  def transformAnnotation(imgWidth: Int, imgHeigth: Int, cropedBox: NormalizedBox,
                          doMirror: Boolean, target: RoiLabel,
                          transformd: ArrayBuffer[NormalizedBox]): Unit = {
    var i = 1
    while (target.bboxes.nElement() > 0 && i <= target.bboxes.size(1)) {
      val resizedBox = NormalizedBox(target.bboxes.valueAt(i, 1),
        target.bboxes.valueAt(i, 2),
        target.bboxes.valueAt(i, 3),
        target.bboxes.valueAt(i, 4))
      if (BboxUtil.meetEmitCenterConstraint(cropedBox, resizedBox)) {
        val transformedBox = new NormalizedBox()
        if (BboxUtil.projectBbox(cropedBox, resizedBox, transformedBox)) {
          if (doMirror) {
            val temp = transformedBox.x1
            transformedBox.x1 = 1 - transformedBox.x2
            transformedBox.x2 = 1 - temp
          }
          transformedBox.setLabel(target.classes.valueAt(1, i))
          transformedBox.setDifficult(target.classes.valueAt(2, i))
          transformd.append(transformedBox)
        }
      }
      i += 1
    }
  }
}


