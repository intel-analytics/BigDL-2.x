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

import com.johnsnowlabs.nlp.AnnotatorModel
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
class PipelinedSparkNLPTransformer(val stages: Array[Transformer])
  extends TextTransformer {

  override def transform(feature: TextFeature): TextFeature = {
    val documentAssembler = new DocumentAssembler().
      setInputCol(TextFeature.text).
      setOutputCol("document")
    val lightPipeline = new LightPipeline(stages = Array(documentAssembler) ++ stages)
    val tokens = lightPipeline.annotate(
      feature.apply[String](TextFeature.text))(
      stages.last.asInstanceOf[AnnotatorModel[_]].getOutputCol).toArray
    // TODO: add key support
    feature(TextFeature.tokens) = tokens
    feature
  }
}

object PipelinedSparkNLPTransformer {
  def apply(stages: Array[Transformer]): PipelinedSparkNLPTransformer = {
    new PipelinedSparkNLPTransformer(stages)
  }
}
