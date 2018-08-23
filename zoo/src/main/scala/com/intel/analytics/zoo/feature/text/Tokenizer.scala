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

/**
 * Transform text to array of tokens.
 * Input key: TextFeature.text
 * Output key: Can be specified by outKey. Default is TextFeature.tokens.
 */
class Tokenizer(override val outKey: String = TextFeature.tokens)
  extends SparkNLPTransformer(outKey) {

  override def labor: AnnotatorModel[_] = {
    new com.johnsnowlabs.nlp.annotator.Tokenizer()
      .setOutputCol("tokens")
  }
}

object Tokenizer {
  def apply(outKey: String = TextFeature.tokens): Tokenizer = {
    new Tokenizer(outKey)
  }
}
