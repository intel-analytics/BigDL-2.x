package com.intel.analytics.zoo.feature.common.persistent.memory

import java.nio.ByteBuffer
import java.util.concurrent.atomic.AtomicInteger

import com.intel.analytics.bigdl.dataset.{ByteRecord, DistributedDataSet}
import com.intel.analytics.bigdl.transform.vision.image.ImageFeature
import com.intel.analytics.bigdl.utils.RandomGenerator
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.EngineRef
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

private[zoo] abstract class NativeArrayConverter[T: ClassTag] extends Serializable {

  def getBytesPerRecord(record: T): Long

  def toArray(recordIterator: Iterator[T],
      countPerPartition: Iterator[(Int, Long)]): Iterator[ArrayLike[T]]
}

private[zoo] class ByteRecordImageConverter extends NativeArrayConverter[ByteRecord] {

  override def getBytesPerRecord(byteRecord: ByteRecord): Long = {
    byteRecord.data.length
  }

  override def toArray(recordIterator: Iterator[ByteRecord],
      countPerPartition: Iterator[(Int, Long)]): Iterator[ArrayLike[ByteRecord]] = {
      val count = countPerPartition.next()
      val optaneDCArray = new NativeVarLenBytesArray(count._1, count._2)
      val labels = ArrayBuffer[Float]()
      var i = 0
      while(recordIterator.hasNext) {
        val data = recordIterator.next()
        val imgBuffer = ByteBuffer.wrap(data.data)
        val width = imgBuffer.getInt
        val height = imgBuffer.getInt
        optaneDCArray.set(i, data.data)
        labels.append(data.label)
        i += 1
      }
      Iterator.single(OptaneDCImageArray(optaneDCArray, labels.toArray))
    }
}

private[zoo] class ImageFeatureConverter extends NativeArrayConverter[ImageFeature] {

  override def getBytesPerRecord(byteRecord: ImageFeature): Long = {
    throw new RuntimeException("Not supported yet")
  }

  override def toArray(recordIterator: Iterator[ImageFeature],
      countPerPartition: Iterator[(Int, Long)]): Iterator[ArrayLike[ImageFeature]] = {
    throw new RuntimeException("Not supported yet")
  }
}

object OptaneDCDataSet {

  private def doRdd[T: ClassTag](data: RDD[T], nativeArrayConverter:
    NativeArrayConverter[T]):
  OptaneDCDataSet[T] = {
    val nodeNumber = EngineRef.getNodeNumber()
    val coaleasedRdd = data.coalesce(nodeNumber, true)
    val countPerPartition = coaleasedRdd.mapPartitions{ iter =>
      require(iter.hasNext)
      var totalBytes: Long = 0L
      var totalRecordNum = 0
      while(iter.hasNext) {
        val record = iter.next()
        totalRecordNum += 1
        totalBytes += nativeArrayConverter.getBytesPerRecord(record)
      }
      Iterator.single((totalRecordNum, totalBytes))
    }
    val arrayRDD = coaleasedRdd.zipPartitions(countPerPartition) {(dataIter, countIter) =>
      nativeArrayConverter.toArray(dataIter, countIter)}.setName("cached with OptaneDC")
      .cache()
    new OptaneDCDataSet[T](arrayRDD.asInstanceOf[RDD[ArrayLike[T]]])
  }

  import scala.reflect.runtime.universe._

  def rdd[T](data: RDD[T])(implicit tag: TypeTag[T]): OptaneDCDataSet[T] = {

    typeOf[T] match {
      case t if t =:= typeOf[ByteRecord] =>
        doRdd[ByteRecord](data.asInstanceOf[RDD[ByteRecord]],
          new ByteRecordImageConverter()).asInstanceOf[OptaneDCDataSet[T]]
      case t if t =:= typeOf[ImageFeature] =>
        doRdd[ImageFeature](data.asInstanceOf[RDD[ImageFeature]],
          new ImageFeatureConverter()).asInstanceOf[OptaneDCDataSet[T]]
    }
  }
}

private[zoo] abstract class ArrayLike[T: ClassTag] extends Serializable {
  def length: Int = throw new Error()

  def apply(i: Int): T = throw new Error()
}

private[zoo] case class OptaneDCImageArray(imgs: NativeVarLenBytesArray, label: Array[Float]) extends ArrayLike[ByteRecord] {
  override def length: Int = {
    imgs.recordNum
  }
  override def apply(i: Int): ByteRecord = {
    // TODO: we may need to change this to Long
    ByteRecord(imgs.get(i), label(i.toInt))
  }
}


/**
 * Wrap a RDD as a DataSet.
 * @param buffer
 */
// T is the returning value type. like ByteRecord
class OptaneDCDataSet[T: ClassTag]
(buffer: RDD[ArrayLike[T]])
  extends DistributedDataSet[T] {

  protected lazy val count: Long = buffer.mapPartitions(iter => {
    require(iter.hasNext)
    val array = iter.next()
    require(!iter.hasNext)
    Iterator.single(array.length)
  }).reduce(_ + _)

  protected var indexes: RDD[Array[Int]] = buffer.mapPartitions(iter => {
    Iterator.single[Array[Int]]((0 until iter.next().length).toArray[Int])
  }).setName("original index").cache()


  override def data(train: Boolean): RDD[T] = {
    val _train = train
    buffer.zipPartitions(indexes)((dataIter, indexIter) => {
      val indexes = indexIter.next()
      val indexOffset = math.max(1, indexes.length)
      val localData = dataIter.next()
      val offset = if (_train) {
        RandomGenerator.RNG.uniform(0, indexOffset).toInt
      } else {
        0
      }
      new Iterator[T] {
        private val _offset = new AtomicInteger(offset)

        override def hasNext: Boolean = {
          if (_train) true else _offset.get() < localData.length
        }

        override def next(): T = {
          val i = _offset.getAndIncrement()
          if (_train) {
            // indexes is an Array, we should improve this
            // as the maximum length is limited by Int.max
            localData(indexes(i % localData.length))
          } else {
            if (i < localData.length) {
              localData(indexes(i))
            } else {
              null.asInstanceOf[T]
            }
          }
        }
      }
    })
  }

  override def size(): Long = count

  override def shuffle(): Unit = {
    indexes.unpersist()
    indexes = buffer.mapPartitions(iter => {
      Iterator.single(RandomGenerator.shuffle((0 until iter.next().length).toArray))
    }).setName("shuffled index").cache()
  }

  override def originRDD(): RDD[_] = buffer

  override def cache(): Unit = {
    buffer.count()
    indexes.count()
    isCached = true
  }

  override def unpersist(): Unit = {
    buffer.unpersist()
    indexes.unpersist()
    isCached = false
  }
}
