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

package com.intel.analytics.zoo.feature.text

import com.johnsnowlabs.nlp.base.{DocumentAssembler, LightPipeline}
import org.apache.spark.ml.Transformer

/**
 * Pipelined transformer that goes through several SparkNLP stages with
 * LightPipeline behind.
 * Input key: TextFeature.text
 * Output key: Specified by the last transformer.
 *
 * @param stages Array of SparkNLPTransformer.
 */
class PipelinedSparkNLPTransformer(val stages: Array[SparkNLPTransformer])
  extends TextTransformer {

  require(stages != null, "stages can't be null")

  override def transform(feature: TextFeature): TextFeature = {
    // DocumentAssembler is the entry point to a SparkNLP pipeline. It creates the first
    // annotation of type Document which may be used by annotators down the road.
    val documentAssembler = new DocumentAssembler()
      .setInputCol(TextFeature.text)
      .setOutputCol("document")
    val NLPstages = stages.map(_.labor)
    NLPstages(0).setInputCols("document")
    var i = 1
    while (i < NLPstages.length) {
      NLPstages(i).setInputCols(NLPstages(i-1).getOutputCol)
      i += 1
    }
    val outputCol = NLPstages.last.getOutputCol
    val outKey = stages.last.outKey
    val lightPipeline = new LightPipeline(stages =
      Array(documentAssembler) ++ NLPstages.map(_.asInstanceOf[Transformer]))
    val res = lightPipeline.annotate(feature[String](TextFeature.text))(outputCol).toArray
    feature(outKey) = res
    feature
  }
}

object PipelinedSparkNLPTransformer {
  def apply(stages: Array[SparkNLPTransformer]): PipelinedSparkNLPTransformer = {
    new PipelinedSparkNLPTransformer(stages)
  }
}
