package com.intel.analytics.zoo.preprocess


import scala.collection.mutable.ArrayBuffer
import scala.collection.immutable.ListMap
import scala.collection.mutable.{Map => MMap}
import scala.io.Source

trait TextProcessing {
  def doTokenize(text: String): List[String] = {
    text.replaceAll("[^a-zA-Z]", " ").toLowerCase().split("\\s+").filter(_.size > 2).toList
  }
  def doStopWords(tokens: List[String], stopWordsCount: Int): List[String] ={
    val topK = ListMap(tokens.groupBy(identity).mapValues(_.length).toSeq.sortBy(-_._2):_*).take(stopWordsCount)
    val newTokens = tokens.filter(!topK.keySet.contains(_))
    newTokens
  }
  def doStopWords(tokens: List[String], stopWords: List[String]): List[String] ={
    tokens.filter(!stopWords.contains(_))
  }

  def doShaping(tokens: List[String], sequenceLength: Int, trunc: String = "pre"): List[String] = {
    val paddedTokens:List[String] = if (tokens.length > sequenceLength) {
      if(trunc=="pre"){
        tokens.take(sequenceLength)
      }else{
        tokens.takeRight(sequenceLength)
      }
    } else {
      tokens ++ List.fill[String](sequenceLength - tokens.length)("")
    }
    paddedTokens
  }

  def doLoadEmbedding(embDir: String):Map[String, List[Float]] = {
    //defult use glove
    val filename = s"$embDir/glove.6B.200d.txt"
    val tokensMapCoefs =  MMap[String, List[Float]]()
    for (line <- Source.fromFile(filename, "ISO-8859-1").getLines) {
      val values = line.split(" ")
      val word = values(0)
      val coefs = values.slice(1, values.length).map(_.toFloat)
      tokensMapCoefs.put(word, coefs.toList)
    }
    tokensMapCoefs.toMap
  }

  def doVectorize(tokens: List[String], embMap:Map[String, List[Float]]): List[List[Float]] = {
    val tokensToCoefs =  ArrayBuffer[List[Float]]()
    for (token <- tokens if token!=""&&embMap.contains(token))
      tokensToCoefs.append(embMap(token))

    if(tokensToCoefs.size < tokens.size){
      val coefZero = List.fill[Float](200)(0)
      val newArray = Array.fill(tokens.size - tokensToCoefs.size)(coefZero)
      tokensToCoefs ++= newArray
    }
    tokensToCoefs.toList
  }

  def doPreprocess(text: String): List[List[Float]] = {
    val tokens = doTokenize(text)
    val shapedTokens = doShaping(doStopWords(tokens,1),500)
    val embMap = doLoadEmbedding(sys.env("EMBEDDING_PATH"))
    val vectorizedTokens = doVectorize(shapedTokens, embMap)
    vectorizedTokens
  }
}
