package com.intel.analytics.zoo.apps.model.inference.spark

import java.io.File

import com.intel.analytics.zoo.apps.textclassfication.processing.TextProcessing
import com.intel.analytics.zoo.pipeline.inference.InferenceSupportive
import com.intel.analytics.zoo.pipeline.inference.JTensor

class TextProcessor (val stopWordsCount: Int, val sequenceLength: Int, val embeddingFilePath: String) extends TextProcessing  with InferenceSupportive with Serializable {
  val wordToIndexMap = doLoadWordToIndexMap(new File(embeddingFilePath))

  def preprocess(text: String): JTensor = {
    val tensor = doPreprocess(text, stopWordsCount, sequenceLength, wordToIndexMap)
    transferTensorToJTensor(tensor)
  }
}
