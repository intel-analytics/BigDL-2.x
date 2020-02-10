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

package com.intel.analytics.zoo.tfpark

import com.intel.analytics.bigdl.dataset.{DistributedDataSet, MiniBatch}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.feature.{DistributedDataSetWrapper, DistributedFeatureSet}
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.EngineRef
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.tensorflow.DataType

import com.intel.analytics.zoo.tfpark.TFTensorNumeric.NumericByteArray


class TFDataFeatureSet(private val graph: Array[Byte],
                       private val initIteratorOp: String,
                       private val outputNames: Array[String],
                       private val outputTypes: Array[DataType],
                       private val dataCount: Int,
                       private val batchSize: Int,
                       private val shardIndex: String)
  extends DistributedFeatureSet[MiniBatch[Float]] {

  private val graphRunnerRDD = getGraphRunnerRDD(graph)

  private def getGraphRunnerRDD(graph: Array[Byte]): RDD[GraphRunner] = {
    val sc = SparkContext.getOrCreate()
    val nodeNumber = EngineRef.getNodeNumber()
    val coreNumber = EngineRef.getCoreNumber()
    // TODO: make sure 1 executor 1 partition

    val broadcastedGraph = sc.broadcast(graph)
    val originRdd = sc.parallelize(
      Array.tabulate(nodeNumber)(_ => "dummy123123"), nodeNumber * 10)
      .mapPartitions(_ => (0 until 200).toIterator)
      .coalesce(nodeNumber)
      .setName("PartitionRDD")
      .persist(StorageLevel.DISK_ONLY)
    originRdd.count()
    val graphRunnerRDD = originRdd.mapPartitions { iter =>
      val graphDef = broadcastedGraph.value
      val runner = GraphRunner(graphDef,
        null, null, null, null, SessionConfig(intraOpParallelismThreads = coreNumber).toByteArray())
      Iterator.single(runner)
    }.setName("GraphRunnerRDD").cache()
    graphRunnerRDD.count()
    graphRunnerRDD
  }
  override def originRDD(): RDD[_] = {
    graphRunnerRDD
  }

  override def data(train: Boolean): RDD[MiniBatch[Float]] = {
    val initOp = this.initIteratorOp
    val names = this.outputNames.toVector
    val types = this.outputTypes.toVector
    val shardIdx = this.shardIndex

    graphRunnerRDD.mapPartitionsWithIndex { case (idx, dataIter) =>
      val graphRunner = dataIter.next()
      def intiIterator(): Unit =  {
        graphRunner.runTargets(Vector(initOp),
          inputs = Vector(Tensor.scalar[Float](idx.toFloat)),
          inputTypes = Vector(DataType.INT64),
          inputNames = Vector(shardIdx))
      }
      if (train) {
        new Iterator[MiniBatch[Float]] {

          override def hasNext(): Boolean = {
            true
          }

          private def getNext() = {
            val outputs = TFDataFeatureSet.generateOutputTensors(types)
            val outputVec = outputs.toVector
            try {
              graphRunner.runOutputs(outputVec, names, types)
            } catch {
              case _: java.lang.IndexOutOfBoundsException =>
                intiIterator()
                graphRunner.runOutputs(outputVec, names, types)
              case _: java.lang.IllegalStateException =>
                intiIterator()
                graphRunner.runOutputs(outputVec, names, types)
              case e: Throwable => throw e
            }
            outputs
          }

          override def next(): MiniBatch[Float] = {
            TFMiniBatch(getNext())
          }
        }
      } else {
        intiIterator()
        new Iterator[MiniBatch[Float]] {

          private var buffer: Array[Tensor[_]] = null
          override def hasNext(): Boolean = {
            if (buffer != null) {
              true
            } else {
              val (success, result) = getNext()
              if (success) {
                buffer = result
              }
              success
            }
          }

          private def getNext() = {
            val outputs = TFDataFeatureSet.generateOutputTensors(types)
            val outputVec = outputs.toVector
            val success = try {
              graphRunner.runOutputs(outputVec, names, types)
              true
            } catch {
              case _: java.lang.IndexOutOfBoundsException => false
              case e: Throwable => throw e
            }
            (success, outputs)
          }

          override def next(): MiniBatch[Float] = {
            if (hasNext()) {
              val result = TFMiniBatch(buffer)
              buffer = null
              result
            } else {
              throw new NoSuchElementException("Next on an empty iterator")
            }
          }
        }
      }
    }
  }

  override def shuffle(): Unit = {

  }

  override def size(): Long = {
    dataCount
  }

  override def toDistributed(): DistributedDataSet[MiniBatch[Float]] = {
    new DistributedDataSetWrapper[MiniBatch[Float]](this)
  }
}

object TFDataFeatureSet {
  def apply(graph: Array[Byte],
            initIteratorOp: String,
            outputNames: Array[String],
            outputTypes: Array[Int],
            dataCount: Int, batchSize: Int, shardIndex: String): TFDataFeatureSet = {
    val types = outputTypes.map(TFUtils.tfenum2datatype)
    new TFDataFeatureSet(graph, initIteratorOp, outputNames, types,
      dataCount, batchSize, shardIndex)
  }

  private def toBatchAll(data: Array[Array[Tensor[_]]],
                         outputTypes: Vector[DataType]): Array[Tensor[_]] = {
    val outputLength = outputTypes.length
    val result = new Array[Tensor[_]](outputLength)
    var i = 0
    while (i < outputLength) {
      if (outputTypes(i) == DataType.STRING) {
        val tensors = new Array[Tensor[Array[Byte]]](data.length)
        var j = 0
        while (j < data.length) {
          tensors(j) = data(j)(i).asInstanceOf[Tensor[Array[Byte]]]
          j += 1
        }
        result(i) = toBatchString(tensors)
      } else {
        val tensors = new Array[Tensor[Float]](data.length)
        var j = 0
        while (j < data.length) {
          tensors(j) = data(j)(i).asInstanceOf[Tensor[Float]]
          j += 1
        }
        result(i) = toBatchFloat(tensors)
      }

      i += 1
    }
    result
  }

  private def toBatchString(data: Array[Tensor[Array[Byte]]]) = {
    val result = Tensor[Array[Byte]](Array(data.length))
    val storage = result.storage().array()
    var i = 0
    while (i < data.length) {
      storage(i) = data(i).value()
      i += 1
    }
    result
  }

  private def toBatchFloat(data: Array[Tensor[Float]]) = {
    val firstDimSize = data.length
    val restDimSize = data(0).size()
    val newShape = Array.tabulate(restDimSize.length + 1) { i =>
      if (i == 0) {
        firstDimSize
      } else {
        restDimSize(i - 1)
      }
    }
    val result = Tensor[Float](newShape)
    var i = 0
    while (i < firstDimSize) {
      result.select(1, i + 1).copy(data(i))
      i += 1
    }
    result
  }

  private def generateOutputTensors(types: Vector[DataType]) = {
    val outputs = Array.tabulate[Tensor[_]](types.length) { i =>
      if (types(i) == DataType.STRING) {
        Tensor[Array[Byte]]()
      } else {
        Tensor[Float]()
      }
    }
    outputs
  }
}
