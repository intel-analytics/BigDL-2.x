package com.intel.analytics.zoo.preprocess

import com.intel.analytics.zoo.pipeline.inference.JTensor

import scala.collection.mutable.{ArrayBuffer, Map => MMap}
import scala.io.Source

import java.util.{List => JList}
import java.lang.{Float => JFloat}
import java.lang.{Integer => JInt}

import scala.collection.JavaConverters._
import collection.JavaConversions._

case class GloveTextProcessor(gloveFilePath: String) extends TextProcessing {


  //for glove.6B only
  override def doLoadEmbedding(embDir: String):Map[String, List[Float]] = {
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

  override def doPreprocess(text: String): List[List[Float]] = {
    val tokens = doTokenize(text)
    val shapedTokens = doShaping(doStopWords(tokens,1),500)
    val embMap = doLoadEmbedding(sys.env("EMBEDDING_PATH"))
    val vectorizedTokens = doVectorize(shapedTokens, embMap)
    vectorizedTokens
  }
}


object preprocessor {

  def main(args: Array[String]): Unit = {
    val textPreprocessor = GloveTextProcessor(sys.env("EMBEDDING_PATH"))
    val text = "It is for for for for test test exwqwq"
    val result = textPreprocessor.doPreprocess(text)
    val tempArray = ArrayBuffer[JList[JFloat]]()
    for (tempList <- result) {
      val javaList = new Array[JFloat](tempList.size)
      for(tempFloat <- tempList) {
        javaList.add(tempFloat.asInstanceOf[JFloat])
      }
      //System.arraycopy(tempList, 0 , javaList , 0 , tempList.size)

      tempArray.add(javaList.toList.asJava)
    }

    val input = tempArray.toArray.toList.asJava
    val data = input.flatten
    val shape = List(input.size().asInstanceOf[JInt],input.get(0).length.asInstanceOf[JInt])
    val tensorInput = new JTensor(data.asJava, shape.asJava)

    print(tensorInput)
  }
}
