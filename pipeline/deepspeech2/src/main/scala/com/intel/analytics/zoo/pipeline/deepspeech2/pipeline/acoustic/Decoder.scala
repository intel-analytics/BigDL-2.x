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

package com.intel.analytics.zoo.pipeline.deepspeech2.pipeline.acoustic

import scala.collection.immutable.HashMap
import scala.collection.mutable.ArrayBuffer

/**
  * Basic decoder class from which all other decoders inherit. Implements several helper functions. Subclasses should implement the decode() method
  * @param alphabet mapping from integers to characters
  * @param blankIndex index for the blank '_' character.
  * @param spaceIndex index for the space ' ' character.
  */
private[intel] abstract class Decoder(
    alphabet: String,
    blankIndex: Int,
    spaceIndex: Int) extends Serializable {

  var alphabetDict: Map[Int, Char] = new HashMap[Int, Char]
  for((c, i) <- alphabet.view.zipWithIndex) {
    alphabetDict += (i -> c)
  }

  /**
   * Given a numeric sequence, returns the corresponding strin    * @param sequence a numeric sequence prepared for the decoding phase
   */
  def convertToString(sequence: Seq[Int]): String = {
    val res = new StringBuilder
    for(x <- sequence){
      res.append(alphabetDict(x))
    }
    res.toString
  }

  /**
   * Given a string, removes blanks and replace space character with space. Option to remove repetitions(e.g. "abbca" -> "abca")
   * @param sequence String
   * @param removeRepetitions boolean, optional: If true, repeating characters are removed. Defaults to false
   * @return
   */
  def processString(sequence: String, removeRepetitions: Boolean = false): String = {

    val res = new StringBuilder
    for((c,i) <- sequence.view.zipWithIndex){
      if(c != alphabetDict(blankIndex)){
        if(!removeRepetitions || i == 0 || c != sequence(i - 1)) {
          res.append(c)
        }
      }
    }
    res.toString
  }

  def getSoftmax(prop:  Array[Array[Float]]): Array[Array[Float]] = {

    val columns = prop.transpose
    columns.map { col =>
      val max = col.max
      val sum = col.map(t => math.exp(t - max).toFloat).sum
      col.map(t => math.exp(t - max).toFloat / sum)
    }.transpose
  }

  /**
    * Given a matrix of character probabilities, returns the decoder's best guess of the transcription
    * @param probs Matrix of character probabilities, where probs[c,t] is the probability of character c at time t
    * @return string: sequence of the model's best guess of the transcript
    */
  def decode(probs: Array[Array[Float]]): String
}


private[intel] class BestPathDecoder (
    alphabet: String,
    blankIndex: Int,
    spaceIndex: Int) extends Decoder(alphabet, blankIndex, spaceIndex) {

  override def decode(probs: Array[Array[Float]]): String = {
    val string: String = convertToString(argMax(probs))
    processString(string, removeRepetitions = true)
  }

  def argMax(probs: Array[Array[Float]]): Seq[Int] = {
    val numSequence = new ArrayBuffer[Int](probs.length)
    probs.transpose.map(prob => numSequence.append(indexOfMax(prob)))
    numSequence.toArray
  }

  def indexOfMax(list: Array[Float]): Int = {
    list.zipWithIndex.maxBy(_._1)._2
  }
}
