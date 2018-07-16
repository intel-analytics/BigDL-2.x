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

import com.esotericsoftware.kryo.{Kryo, KryoSerializable}
import com.esotericsoftware.kryo.io.{Output, Input => KyroInput}
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.nn.{LookupTable, AddConstant => TAddConstant, Sequential => TSequential}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.api.keras.layers.EmbeddingGloVe.WeightHolder
import org.apache.commons.lang.SerializationUtils
import org.apache.spark.utils.SparkUtils
import org.slf4j.LoggerFactory

import scala.collection.mutable
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

  if (wordIndex != null) {
    require(! wordIndex.values.exists(_ == 0),
      "In wordIndex, index should start from 1 with 0 reserved for unknown words.")
  }

  private val embeddingFile: String = gloveDir + "/" + embeddingType + ".txt"

  private val embeddingMatrix: WeightHolder[T] =
    new WeightHolder[T](buildEmbeddingMatrix(), embeddingType + getName())

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (Array(), Array())
  }

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
    for (i <- 1 until inputDim) {
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
    layer.setWeightsBias(Array(embeddingMatrix.weight))
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

  private def isDriver = try {
    SparkUtils.isDriver
  } catch {
    case e: NullPointerException =>
      true
  }

  private val logger = LoggerFactory.getLogger(getClass)

  private val weightRegistry = new mutable.WeakHashMap[String, Array[Byte]]()

  private[zoo] def getWeightRegistrySize = weightRegistry.size

  class WeightHolder[T: ClassTag](
      @transient var weight: Tensor[T],
      private var id: String)(implicit ev: TensorNumeric[T])
    extends Serializable with KryoSerializable {

    private trait CommonOutputStream {
      def writeInt(value: Int): Unit
      def write(value: Array[Byte]): Unit
      def writeString(value: String): Unit
    }

    private trait CommonInputStream {
      def readInt(): Int
      def read(buff: Array[Byte], off: Int, len: Int): Int
      def skip(len: Int): Unit
      def readString(): String
    }

    private def writeInternal(out: CommonOutputStream): Unit = {
      val (w, _) = getOrCreateWeight(id) {
        SerializationUtils.serialize(weight)
      }
      val len = w.length
      out.writeString(id)
      if (isDriver) {
        out.writeInt(len)
        out.write(w)
      } else {
        out.writeInt(0)
      }
    }

    private def readInternal(in: CommonInputStream): Unit = {
      id = in.readString()
      val (w, isCreated) = getOrCreateWeight(id) {
        val len = in.readInt()
        require(len != 0, "Weight length should not be zero, " +
          "please set logging level to debug for more information")
        val weight = new Array[Byte](len)
        var numOfBytes = 0
        while (numOfBytes < len) {
          val read = in.read(weight, numOfBytes, len - numOfBytes)
          numOfBytes += read
        }
        weight
      }

      if (!isCreated) {
        val len = in.readInt()
        in.skip(len)
      }

      weight = SerializationUtils.deserialize(w).asInstanceOf[Tensor[T]]
    }

    private def writeObject(out: java.io.ObjectOutputStream): Unit = {
      writeInternal(new CommonOutputStream {
        override def writeInt(value: Int): Unit = out.writeInt(value)

        override def write(value: Array[Byte]): Unit = out.write(value)

        override def writeString(str: String): Unit = out.writeUTF(str)
      })
    }

    private def readObject(in: java.io.ObjectInputStream): Unit = {
      readInternal(new CommonInputStream {
        override def read(buff: Array[Byte], off: Int, len: Int): Int = in.read(buff, off, len)

        override def skip(len: Int): Unit = in.skip(len)

        override def readInt(): Int = in.readInt()

        override def readString(): String = in.readUTF()
      })
    }

    override def read(kryo: Kryo, in: KyroInput): Unit = {
      readInternal(new CommonInputStream {
        override def read(buff: Array[Byte], off: Int, len: Int): Int = in.read(buff, off, len)

        override def skip(len: Int): Unit = in.skip(len)

        override def readInt(): Int = in.readInt()

        override def readString(): String = in.readString()
      })
    }

    override def write(kryo: Kryo, out: Output): Unit = {
      writeInternal(new CommonOutputStream {
        override def writeInt(value: Int): Unit = out.writeInt(value)

        override def write(value: Array[Byte]): Unit = out.write(value)

        override def writeString(value: String): Unit = out.writeString(value)
      })
    }
  }

  // return (weight, isCreated)
  private def getOrCreateWeight(id: String)
     (createWeight: => Array[Byte]): (Array[Byte], Boolean) = {
    if (weightRegistry.contains(id)) {
      logger.debug(s"weight for embedding: $id already exists, read from registry. " +
        s"Registry size: $getWeightRegistrySize")
      (weightRegistry(id), false)
    } else {
      this.synchronized {
        if (weightRegistry.contains(id)) {
          logger.debug(s"weightM for embedding: $id already exists, read from registry. " +
            s"Registry size: $getWeightRegistrySize")
          (weightRegistry(id), false)
        } else {
          logger.debug(s"weight for embedding: $id does not exist, created it. " +
            s"Registry size: $getWeightRegistrySize")
          val weight = createWeight
          weightRegistry.put(id, weight)
          (weight, true)
        }
      }
    }
  }
}
