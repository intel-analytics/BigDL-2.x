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

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.{FeatureTransformer, ImageFeature}

@SerialVersionUID(-3940915022534165986L)
class LabelOutput(labelMap: Map[Int, String], clses : String, probs : String)
  extends FeatureTransformer{
  override def transformMat(imageFeature: ImageFeature): Unit = {
    val predictOutput = imageFeature[Tensor[Float]](ImageFeature.predict)
    val total = predictOutput.nElement()
    val start = predictOutput.storageOffset() - 1
    val end = predictOutput.storageOffset() - 1 + predictOutput.nElement()
    val clsNo = end - start
    val sortedResult = predictOutput.storage().array().slice(start, end).
      zipWithIndex.sortWith(_._1 > _._1).toList.toArray

    val classes: Array[String] = new Array[String](clsNo)
    val probilities  : Array[Float] = new Array[Float](clsNo)

    var index = 0;

    while (index < clsNo) {
      val clsName = labelMap(sortedResult(index)._2)
      val prob = sortedResult(index)._1
      classes(index) = clsName
      probilities(index) = prob
      index += 1
    }

    imageFeature(clses) = classes
    imageFeature(probs) = probilities
  }
}

object LabelOutput {
  def apply(labelMap: Map[Int, String], classes: String, probs: String): LabelOutput =
    new LabelOutput(labelMap, classes, probs)
}
