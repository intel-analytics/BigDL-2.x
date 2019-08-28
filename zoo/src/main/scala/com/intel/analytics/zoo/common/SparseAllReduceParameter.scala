package com.intel.analytics.zoo.common

import java.util.concurrent.{Callable, Future}

import com.intel.analytics.bigdl.parameters.AllReduceParameter._
import com.intel.analytics.bigdl.parameters._
import com.intel.analytics.bigdl.tensor.{IndexedSlicesTensor, SparseTensorUtils, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import org.apache.commons.lang.exception.ExceptionUtils
import org.apache.log4j.Logger
import org.apache.spark.TaskContext
import org.apache.spark.sparkExtension.SparkExtension
import org.apache.spark.storage.{BlockId, BlockManagerWrapper, StorageLevel}

import scala.collection.JavaConverters._
import scala.reflect._

class SparseAllReduceParameter[U: ClassTag](partitionNum: Int,
                               val shape: Array[Int])(implicit ev: TensorNumeric[U]) {
  @transient private var taskSize = 0
  @transient private var extraSize = 0
  @transient private var partitionId: Int = 0

  /** Tensor to hold a slice of the global weights. */
  @transient lazy val weightPartition: Tensor[U] = readWeightPartition()

  /** Tensor to hold a slice of the global gradients. */
  @transient lazy val gradientPartition: IndexedSlicesTensor[U] = readGradientPartition()

  private def readObject(in: java.io.ObjectInputStream): Unit = {
    in.defaultReadObject()
    taskSize = shape.head / partitionNum
    extraSize = shape.head % partitionNum
    partitionId = TaskContext.getPartitionId()
  }

  private def readWeightPartition(): Tensor[U] = {
    val blockId = getWeightPartitionId()
    BlockManagerWrapper.getLocal(blockId)
      .map(_.data.next().asInstanceOf[Tensor[U]])
      .getOrElse(throw new IllegalStateException("Please initialize first!"))
  }

  private def readGradientPartition(): IndexedSlicesTensor[U] = {
    // return gradient partition in blockmanager or empty if it doesn't exist
    val blockId = getGradientPartitionId()
    BlockManagerWrapper.getLocal(blockId)
      .map(_.data.next().asInstanceOf[IndexedSlicesTensor[U]])
      .getOrElse(IndexedSlicesTensor[U]())
  }

  def init(parameter: Tensor[U])(implicit ev: TensorNumeric[U]):
  (Int, Int, Int) = {
    require(parameter.dim() == 2, "Only support 2 dim tensor")
    val _classTag = classTag[U]
    // start of first dim
    val start = partitionId * taskSize + math.min(partitionId, extraSize)
    val length = taskSize + (if (partitionId < extraSize) 1 else 0)

    val _weights = Tensor[U](length)(_classTag, ev).copy(parameter.narrow(1, start + 1, length))
    // DO NOT NEED PUT gradientPartition since every time it's different
//    val _gradients = Tensor[U](length)(_classTag, ev)

    BlockManagerWrapper.removeBlock(getWeightPartitionId())
    BlockManagerWrapper.putSingle(getWeightPartitionId(),
      _weights, StorageLevel.MEMORY_AND_DISK, tellMaster = false)
//    BlockManagerWrapper.removeBlock(getGradientPartitionId())
//    BlockManagerWrapper.putSingle(getGradientPartitionId(),
//      _gradients, StorageLevel.MEMORY_AND_DISK, tellMaster = false)
    val blockId = getWeightBlockId(partitionId)
    val fp16param = new FP16CompressedTensorWrapper[U].apply(length)
    fp16param.compress(0, parameter, start, length)
    BlockManagerWrapper.putBytes(blockId, fp16param.bytes(), StorageLevel.MEMORY_ONLY_SER)
    (partitionId, start, length)
  }

  private def getWeightBlockId(pid: Int): BlockId = {
    SparkExtension.getLocalBlockId("sweightBytes" + pid)
  }

  private def getWeightPartitionId(): BlockId = {
    SparkExtension.getLocalBlockId("sweights" + partitionId)
  }

  private def getGradientPartitionId(): BlockId = {
    SparkExtension.getLocalBlockId("sgradients" + partitionId)
  }

  private def getGradientIndiceBlockId(pidFrom: Int, pidTo: Int): BlockId = {
    SparkExtension.getLocalBlockId(pidTo + "gradientIndiceBytes" + pidFrom)
  }

  private def getGradientValueBlockId(pidFrom: Int, pidTo: Int): BlockId = {
    SparkExtension.getLocalBlockId(pidTo + "gradientValueBytes" + pidFrom)
  }

  /**
   * Use a fixed thread pool to launch a thread for each partition of the weights. Each thread
   * requests a partition of the weights from the Spark block manager and copies it into
   * `localParameter`.
   *
   * @param localParameter The Tensor that will hold the retrieved weights.
   * @return A [[FutureResult]] which contains a [[Future]] for each thread.
   */
  def getWeights(localParameter: Tensor[U]): FutureResult[Int] = {
    val par = localParameter.resize(localParameter.nElement())
    val taskSize2 = localParameter.nElement() / partitionNum
    val extraSize2 = localParameter.nElement() % partitionNum

    val tasks = (0 until partitionNum).map { pid =>
      AllReduceParameter.syncPool.submit {
        new Callable[Int] {
          override def call(): Int = {
            try {
              val blockId = getWeightBlockId(pid)
              val localBuffer = BlockManagerWrapper.getLocalOrRemoteBytes(blockId).getOrElse {
                throw new RuntimeException(s"Didn't find weight block $blockId in the block " +
                  s"manager. Did you initialize this AllReduceParameter on every executor?")
              }
              val start = pid * taskSize2 + math.min(pid, extraSize2)
              val length = taskSize2 + (if (pid < extraSize2) 1 else 0)

              require(localBuffer.array().length == length * 2)
              SerializerInstance.create(localBuffer).deCompress(0, par, start, length)
              BlockManagerWrapper.unlock(blockId)
              pid
            } catch {
              case t: Throwable =>
                logger.error("Error: " + ExceptionUtils.getStackTrace(t))
                throw t
            }
          }
        }
      }
    }
    new FutureResult(tasks)
  }

  def aggregateGradientPartition(avgNumbers: Int): Unit = {
    require(partitionId < partitionNum, s"This parameter was created with $partitionNum " +
      s"partitions. It cannot be used on RDDs with > $partitionNum partitions.")
    val indiceParams = new Array[CompressedTensor[U]](partitionNum)
    val valueParams = new Array[CompressedTensor[U]](partitionNum)
    val sgThreads = (0 until partitionNum).map { pid =>
      new Callable[Int] {
        override def call(): Int = {
          try {
            val indiceBlockId = getGradientIndiceBlockId(pid, partitionId)
            val tmp = BlockManagerWrapper.getLocalOrRemoteBytes(indiceBlockId).get
            indiceParams(pid) = SerializerInstance.create(tmp)
            BlockManagerWrapper.unlock(indiceBlockId)

            val valueBlockId = getGradientValueBlockId(pid, partitionId)
            val tmp2 = BlockManagerWrapper.getLocalOrRemoteBytes(valueBlockId).get
            valueParams(pid) = SerializerInstance.create(tmp2)
            BlockManagerWrapper.unlock(valueBlockId)
            pid
          } catch {
            case t: Throwable =>
              logger.error("Error: " + ExceptionUtils.getStackTrace(t))
              throw t
          }
        }
      }
    }
    AllReduceParameter.syncPool.invokeAll(sgThreads.asJava)

    // TODO: Decompress the gradients and add them to gradientPartition
    val gradients = indiceParams.zip(valueParams).map { case (indice, value) =>
      val indiceTensor = Tensor[U]()
      val valueTensor = Tensor[U]()
      indice.deCompress(indiceTensor)
      value.deCompress(valueTensor)
      IndexedSlicesTensor(indiceTensor.asInstanceOf[Tensor[Int]], valueTensor, shape)
    }
    var aggregatedG = gradients(0)
    for (i <- 1 until gradients.length) {
      aggregatedG = SparseTensorUtils.addSparseTensor[U](aggregatedG, gradients(i))
        .asInstanceOf[IndexedSlicesTensor[U]]
    }
    SparseTensorUtils.copySparseTensor(aggregatedG, gradientPartition)

    gradientPartition.div(ev.fromType(avgNumbers))
  }

  def putGradients(parameter: Tensor[U]): Unit = {
    val _classTag = classTag[U]
    val tensor = parameter.asInstanceOf[IndexedSlicesTensor[U]]

    // generate start/length
    val starts = new Array[Int](partitionNum)
    starts(0) = 0
    val lens = new Array[Int](partitionNum)

    val interval = tensor._shape.head / partitionNum
    var i = 1
    var j = 0
    while (i < partitionNum) {
      while (j < tensor._indices.length && tensor._indices(j) < interval * i) {
        j += 1
      }
      starts(i) = j
      lens(i - 1) = starts(i) - starts(i - 1)
      i += 1
    }
    lens(i - 1) = tensor._shape.head - starts.last


    // split indexedSlicesTensor into 2 dense tensors
    AllReduceParameter.computePool.invokeAll((0 until partitionNum).map(i =>
      new Callable[Int] {
        override def call(): Int = {
          val indiceBlockId = getGradientIndiceBlockId(partitionId, i)
          val indiceBlock = BlockManagerWrapper.getLocalBytes(indiceBlockId)
          if (indiceBlock.isDefined) {
            val fp16param = new FP16CompressedTensorWrapper[Int].apply(indiceBlock.get)
            fp16param.compress(0, Tensor(tensor._indices, Array(tensor._indices.length)),
              starts(i), lens(i - 1))
            i
          } else {
            val fp16param = new FP16CompressedTensorWrapper[Int].apply(lens(i - 1))
            fp16param.compress(0, Tensor(tensor._indices, Array(tensor._indices.length)),
              starts(i), lens(i - 1))
            BlockManagerWrapper.putBytes(indiceBlockId, fp16param.bytes(), StorageLevel.MEMORY_ONLY_SER)
            i
          }

          val valueBlockId = getGradientValueBlockId(partitionId, i)
          val valueBlock = BlockManagerWrapper.getLocalBytes(valueBlockId)
          val valueTensor = Tensor[U](tensor._values.slice(starts(i), lens(i)).flatten,
            Array(lens(i), tensor._shape.last))
          if (valueBlock.isDefined) {
            val fp16param = new FP16CompressedTensorWrapper[U].apply(valueBlock.get)
            fp16param.compress(valueTensor)
            i
          } else {
            val fp16param = new FP16CompressedTensorWrapper[U].apply(lens(i))
            fp16param.compress(valueTensor)
            BlockManagerWrapper.putBytes(indiceBlockId, fp16param.bytes(), StorageLevel.MEMORY_ONLY_SER)
            i
          }
        }
      }
    ).asJava)
  }

  /**
    * Put the portion of the weights that this partition is responsible for to the block manager.
    * Weights are placed locally, then pulled when needed by other partitions.
    */
  def sendWeightPartition(): Unit = {
    val blockId = getWeightBlockId(partitionId)
    val localBuffer = BlockManagerWrapper.getLocalBytes(blockId).getOrElse {
      throw new RuntimeException(s"Didn't find weight block $blockId in the block " +
        s"manager. Did you initialize this AllReduceParameter on every executor?")
    }
    SerializerInstance.create(localBuffer).compress(weightPartition)

    val weightsId = getWeightPartitionId()
    val weights = BlockManagerWrapper.getLocal(weightsId)
      .map(_.data.next().asInstanceOf[Tensor[U]])
      .getOrElse(throw new IllegalStateException("Please initialize AllReduceParameter first!"))
    weights.copy(weightPartition)
  }
}

object SparseAllReduceParameter {
  val logger: Logger = Logger.getLogger(getClass)
}

class FutureResult[T](private val futures: Seq[Future[T]]) {
  def waitResult(): Seq[T] = futures.map(_.get())
}