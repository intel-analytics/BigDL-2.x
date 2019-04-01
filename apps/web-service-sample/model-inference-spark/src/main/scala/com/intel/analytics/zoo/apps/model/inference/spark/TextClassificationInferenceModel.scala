package com.intel.analytics.zoo.apps.model.inference.spark

import com.intel.analytics.zoo.pipeline.inference.{InferenceModel, JTensor}

class TextClassificationInferenceModel(val supportedConcurrentNum: Int, val stopWordsCount: Int, val sequenceLength: Int, val embeddingFilePath: String)
  extends InferenceModel(supportedConcurrentNum) with Serializable {


  val textProcessor = new TextProcessor(stopWordsCount, sequenceLength, embeddingFilePath)

  def preprocess(text: String): JTensor = textProcessor.preprocess(text)
}