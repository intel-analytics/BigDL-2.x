package com.intel.analytics.zoo.tfpark

import com.intel.analytics.bigdl.dataset.{DistributedDataSet, MiniBatch}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.feature.{DistributedDataSetWrapper, DistributedFeatureSet}
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.EngineRef
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel


class TFDataFeatureSet(graph: Array[Byte],
                                    private val trainInitIteratorOp: String,
                                    private val validationInitIterationOp: String,
                                    private val trainOutputNames: Array[String],
                                    private val validationOutputNames: Array[String])
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
      .mapPartitions(_ => (0 until 20000000).toIterator)
      .coalesce(nodeNumber)
      .setName("PartitionRDD")
      .persist(StorageLevel.DISK_ONLY)
    originRdd.count()
    val graphRunnerRDD = originRdd.mapPartitions { iter =>
      val graphDef = broadcastedGraph.value
      val runner = new GraphRunner(graphDef,
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
    val initOp = this.trainInitIteratorOp
    val outputNames = this.trainOutputNames.toVector
    graphRunnerRDD.mapPartitions{dataIter =>
      val graphRunner = dataIter.next()

      new Iterator[MiniBatch[Float]] {

        private var buffer: Array[Tensor[Float]] = null
        override def hasNext(): Boolean = {
          if (buffer == null) {
            val result = getNext(train)
            buffer = result._2
            result._1
          } else {
            true
          }

        }

        private def getNext(restart: Boolean) = {
          val outputs = Array.tabulate(outputNames.length)(_ => Tensor[Float]())
          val outputVec = outputs.toVector
          val success = try {
            graphRunner.runOutputs(outputVec, outputNames)
            true
          } catch {
            case _: java.lang.IndexOutOfBoundsException =>
              if (restart) {
                graphRunner.runTargets(Vector(initOp))
                graphRunner.runOutputs(outputVec, outputNames)
                true
              } else {
                false
              }
            case _: java.lang.IllegalArgumentException =>
              graphRunner.runTargets(Vector(initOp))
              graphRunner.runOutputs(outputVec, outputNames)
              true
            case e: _ => throw e
          }
          (success, outputs)
        }

        override def next(): MiniBatch[Float] = {
          if (hasNext()) {
            val miniBatch = MiniBatch(buffer)
            buffer = null
            miniBatch
          } else {
            throw new NoSuchElementException()
          }
        }
      }
    }

  }

  override def shuffle(): Unit = {

  }

  override def size(): Long = {
    data(false).count()
  }

  override def toDistributed(): DistributedDataSet[MiniBatch[Float]] = {
    new DistributedDataSetWrapper[MiniBatch[Float]](this)
  }
}
