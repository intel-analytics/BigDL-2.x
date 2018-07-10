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

package com.intel.analytics.zoo.pipeline.api.keras.layers

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.nn.{LookupTable, AddConstant => TAddConstant, Sequential => TSequential}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.pipeline.api.Net

import scala.collection.mutable.{Map => MMap}
import scala.io.Source
import scala.reflect.ClassTag

/**
 * Turn positive integers (indexes) into dense vectors of fixed size.
 * The input of this layer should be 2D.
 *
 * This layer can only be used as the first layer in a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 * // TODO: add docs
 * @param inputShape A Single Shape, does not include the batch dimension.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class EmbeddingGloVe[T: ClassTag] private(
    val embeddingType: String = "glove.6B.200d",
    val gloveDir: String,
    val wordIndex: Map[String, Int] = null,
    val trainable: Boolean = false,
    override val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends Embedding[T](EmbeddingGloVe.calcInputDimFromTokenIndex(wordIndex),
    EmbeddingGloVe.getOutputDimFromGlove(embeddingType), null, null, inputShape) with Net {

  private val embeddingFile: String = gloveDir + "/" + embeddingType + ".txt"

  def prepareGlove(): Map[Int, Array[T]] = {
    // TODO: refactor when embeddingType == null
    println("Indexing word vectors.")
    val indexVec = MMap[Int, Array[T]]()
    // TODO: file path or download?
    for (line <- Source.fromFile(embeddingFile, "ISO-8859-1").getLines) {
      val values = line.split(" ")
      val word = values(0)
      if (wordIndex.keySet.contains(word)) {
        val vector = values.slice(1, values.length).map(v => ev.fromType(v.toFloat))
        indexVec.put(wordIndex(word), vector)
      }
    }
    println(s"Found ${indexVec.size} word vectors.")
    indexVec.toMap
  }

  def buildEmbeddingMatrix(): Tensor[T] = {
    val indexVec = prepareGlove()
    val weights = Tensor[T](inputDim, outputDim).zero()
    for (i <- 0 until (inputDim -1)) {
      if (indexVec.get(i).isDefined) {
        weights.narrow(1, i + 1, 1).copy(Tensor[T](indexVec(i), Array(outputDim)))
      }
    }
    weights
  }

  def getTokenIndex: Map[String, Int] = {
    // TODO: wordIndex=null construct the whole glove
    wordIndex
  }

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val model = TSequential[T]()
    model.add(TAddConstant(1.0))
    val layer = LookupTable(
      nIndex = inputDim,
      nOutput = outputDim,
      wRegularizer = wRegularizer)
    layer.setWeightsBias(Array(buildEmbeddingMatrix()))
    model.add(layer)
    if (!trainable) model.freeze(layer.getName)
    model.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }

}

object EmbeddingGloVe {
  def apply[@specialized(Float, Double) T: ClassTag](
      embeddingType: String = "glove.6B.200d",
      gloveDir: String,
      wordIndex: Map[String, Int] = null,
      trainable: Boolean = false,
      inputShape: Shape = null)(implicit ev: TensorNumeric[T]): EmbeddingGloVe[T] = {
    new EmbeddingGloVe[T](embeddingType, gloveDir, wordIndex, trainable, inputShape)
  }

  def calcInputDimFromTokenIndex(tokenIndex: Map[String, Int]): Int = {
    // Use max here in case the indices are not continuous.
    // +1 for unknown index 0.
    tokenIndex.values.max + 1
  }

  def getOutputDimFromGlove(embeddingType: String): Int = {
    require(embeddingType != null, "Embedding type cannot be null")
    embeddingType match {
      case "glove.6B.50d" => 50
      case "glove.6B.100d" => 100
      case "glove.6B.200d" => 200
      case "glove.6B.300d" => 300
      case _ => throw new IllegalArgumentException(s"Unsupported embedding type: " +
        s"$embeddingType")
    }
  }
}
