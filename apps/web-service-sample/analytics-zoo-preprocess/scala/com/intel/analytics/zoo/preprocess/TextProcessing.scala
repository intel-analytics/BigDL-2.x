package com.intel.analytics.zoo.preprocess

import java.lang.{Float => JFloat}
import java.util.{List => JList, Map => JMap}

import scala.collection.immutable.ListMap
import scala.collection.mutable.{ArrayBuffer, Map => MMap}
import scala.io.Source

trait TextProcessing {
  def doTokenize(text: String): List[String] = {
    text.replaceAll("[^a-zA-Z]", " ").toLowerCase().split("\\s+").filter(_.size > 2).toList
  }

  def doStopWords(tokens: List[String], stopWordsCount: Int): List[String] = {
    val topK = ListMap(tokens.groupBy(identity).mapValues(_.length).toSeq.sortBy(-_._2): _*).take(stopWordsCount)
    val newTokens = tokens.filter(!topK.keySet.contains(_))
    newTokens
  }

  def doStopWords(tokens: List[String], stopWords: List[String]): List[String] = {
    tokens.filter(!stopWords.contains(_))
  }

  def doShaping(tokens: List[String], sequenceLength: Int, trunc: String = "pre"): List[String] = {
    val paddedTokens: List[String] = if (tokens.length > sequenceLength) {
      if (trunc == "pre") {
        tokens.take(sequenceLength)
      } else {
        tokens.takeRight(sequenceLength)
      }
    } else {
      tokens ++ List.fill[String](sequenceLength - tokens.length)("")
    }
    paddedTokens
  }

  def doLoadEmbedding(embFilePath: String): Map[String, List[Float]] = {
    //defult use as glove
    val tokensMapCoefs = MMap[String, List[Float]]()
    for (line <- Source.fromFile(embFilePath, "ISO-8859-1").getLines) {
      val values = line.split(" ")
      val word = values(0)
      val coefs = values.slice(1, values.length).map(_.toFloat)
      tokensMapCoefs.put(word, coefs.toList)
    }
    tokensMapCoefs.toMap
  }

  def doVectorize(tokens: List[String], embMap: Map[String, List[Float]]): List[List[Float]] = {
    val tokensToCoefs = ArrayBuffer[List[Float]]()
    for (token <- tokens if token != "" && embMap.contains(token)) {
      tokensToCoefs.append(embMap(token))
    }
    val coefSize = tokensToCoefs(0).length
    if (tokensToCoefs.size < tokens.size) {
      val coefZero = List.fill[Float](coefSize)(0)
      val newArray = Array.fill(tokens.size - tokensToCoefs.size)(coefZero)
      tokensToCoefs ++= newArray
    }
    tokensToCoefs.toList
  }

  def doPreprocess(text: String, stopWordsCount: Int, sequenceLength: Int): List[List[Float]] = {
    val tokens = doTokenize(text)
    val shapedTokens = doShaping(doStopWords(tokens, stopWordsCount), sequenceLength)
    val embMap = doLoadEmbedding(sys.env("EMBEDDING_PATH"))
    val vectorizedTokens = doVectorize(shapedTokens, embMap)
    vectorizedTokens
  }

  def doVectorizeWithJMap(tokens: List[String], embMap: JMap[String, JList[JFloat]]): List[JList[JFloat]] = {
    val tokensToCoefs = ArrayBuffer[JList[JFloat]]()
    for (token <- tokens if token != "" && embMap.containsKey(token)) {
      tokensToCoefs.append(embMap.get(token))
    }
    val coefSize = tokensToCoefs(0).size()
    if (tokensToCoefs.size < tokens.size) {
      val coefZero = List.fill[JFloat](coefSize)(float2Float(0)).asInstanceOf[JList[JFloat]]
      val newArray = Array.fill(tokens.size - tokensToCoefs.size)(coefZero)
      tokensToCoefs ++= newArray
    }
    tokensToCoefs.toList
  }

  def doPreprocessWithEmbMap(text: String, stopWordsCount: Int, sequenceLength: Int, embMap: JMap[String, JList[JFloat]]): List[JList[JFloat]] = {
    val tokens = doTokenize(text)
    val shapedTokens = doShaping(doStopWords(tokens, stopWordsCount), sequenceLength)
    val vectorizedTokens = doVectorizeWithJMap(shapedTokens, embMap)
    vectorizedTokens
  }
}