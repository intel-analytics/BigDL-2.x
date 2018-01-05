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

package com.intel.analytics.zoo.pipeline.common.caffe

import com.google.protobuf.GeneratedMessage
import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.nn.Proposal
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import pipeline.caffe.Caffe.LayerParameter

import scala.util.parsing.json.JSON


class PythonConverter(implicit ev: TensorNumeric[Float]) extends Customizable[Float] {


  override def convertor(layer: GeneratedMessage): Seq[ModuleNode[Float]] = {
    val param = layer.asInstanceOf[LayerParameter].getPythonParam
    val layerName = param.getLayer
    layerName match {
      case "ProposalLayer" => convertProposal(layer)
      case "AnchorTargetLayer" =>
        null
      case "ProposalTargetLayer" =>
        null
    }
  }

  def convertProposal(layer: GeneratedMessage): Seq[ModuleNode[Float]] = {
    val param = layer.asInstanceOf[LayerParameter].getPythonParam
    val paramStr = param.getParamStr.replaceAll("'", "\"")
    val (ratios, scales) = JSON.parseFull(paramStr) match {
      case Some(map: Map[String, Any]) =>
        val ratios = map("ratios").asInstanceOf[List[Double]].toArray.map(_.toFloat)
        val scales = map("scales").asInstanceOf[List[Double]].toArray.map(_.toFloat)
        (ratios, scales)
      case _ =>
        val ratios = Array[Float](0.5f, 1.0f, 2.0f)
        val scales = Array[Float](8, 16, 32)
        (ratios, scales)
    }
    // for faster rcnn
    val (preNmsTopN, postNmsTopN) = if (ratios.length == 3) {
      // vgg
      (6000, 300)
    } else {
      (12000, 200)
    }
    Seq(Proposal(preNmsTopN, postNmsTopN, ratios, scales)
      .setName(getLayerName(layer)).inputs())
  }
}
