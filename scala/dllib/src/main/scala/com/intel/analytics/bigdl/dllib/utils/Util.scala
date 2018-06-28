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

package com.intel.analytics.bigdl.utils

import java.io._

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.Container
import com.intel.analytics.bigdl.nn.tf.Const
import com.intel.analytics.bigdl.tensor.TensorNumericMath.{NumericWildcard, TensorNumeric}
import com.intel.analytics.bigdl.tensor._
import org.apache.commons.lang3.SerializationException

import scala.reflect.ClassTag
import scala.util.Try

object Util {
  def kthLargest(arr: Array[Long], l: Int, r: Int, k: Int): Long = {
    if (k == 0) return Long.MaxValue
    val pos = randomPartition(arr, l, r)
    if (pos-l == k-1)  return arr(pos)

    if (pos-l > k-1) return kthLargest(arr, l, pos-1, k)

    kthLargest(arr, pos + 1, r, k - pos + l - 1)
  }

  def swap(arr: Array[Long], i: Int, j: Int): Unit = {
    val temp = arr(i)
    arr(i) = arr(j)
    arr(j) = temp
  }

  private def partition(arr: Array[Long], l: Int, r: Int): Int = {
    val x = arr(r)
    var i = l
    for (j <- l to (r - 1)) {
      if (arr(j) > x) {
        swap(arr, i, j);
        i += 1
      }
    }
    swap(arr, i, r);
    i
  }

  private def randomPartition(arr: Array[Long], l: Int, r: Int): Int = {
    val n = r - l + 1;
    val pivot = ((Math.random()) % n).toInt;
    swap(arr, l + pivot, r);
    partition(arr, l, r);
  }

  private[bigdl] def shift[B](data : Array[B], from : Int, to : Int): Array[B] = {
    require(from < data.length && from >= 0, s"invalid from $from array length is ${data.length}")
    require(to < data.length && to >= 0, s"invalid to $to array length is ${data.length}")
    if (from == to) {
      data
    } else if (from < to) {
      var i = from
      while(i < to) {
        val tmp = data(i)
        data(i) = data(i + 1)
        data(i + 1) = tmp
        i += 1
      }
      data
    } else {
      var i = from
      while(i > to) {
        val tmp = data(i)
        data(i) = data(i - 1)
        data(i - 1) = tmp
        i -= 1
      }
      data
    }
  }


  private[bigdl] def getAndClearWeightBias[T: ClassTag]
  (parameters: (Array[Tensor[T]], Array[Tensor[T]]))(implicit ev: TensorNumeric[T])
  : Array[Tensor[T]] = {
    if (parameters._1.length != 0) {
      var i = 0
      val weightsBias = new Array[Tensor[T]](parameters._1.length)
      val isQuantized = parameters._1.exists(_.getTensorType == QuantizedType)
      val (isCompacted, storage) = if (!isQuantized) {
        val storage = Storage(parameters._1(0).storage.array())
        (parameters._1.map(_.nElement()).sum == storage.length(), storage)
      } else {
        (false, null)
      }

      // get weight and bias
      while (i < parameters._1.length) {
        if (parameters._1(i) != null) {
          val wb = parameters._1(i)
          wb.getTensorType match {
            case QuantizedType =>
              val quantTensor = wb.asInstanceOf[QuantizedTensor[T]]
              weightsBias(i) = QuantizedTensor[T](quantTensor.getStorage, quantTensor.maxOfRow,
                quantTensor.minOfRow, quantTensor.sumOfRow, quantTensor.size(), quantTensor.params)
            case _ =>
              weightsBias(i) = if (isCompacted) {
                Tensor[T](storage, wb.storageOffset(), wb.size(), wb.stride())
              } else {
                Tensor[T](Storage(wb.storage().array()), wb.storageOffset(), wb.size(), wb.stride())
              }
          }
          i += 1
        }
      }
      // clear parameters
      clearTensor(parameters._1)
      clearTensor(parameters._2)

      weightsBias
    } else {
      // just return an empty array when parameters is empty.
      Array()
    }
  }

  private[bigdl] def getAndClearConsts[T: ClassTag](
        model: Container[_, _, T])(implicit ev: TensorNumeric[T]): Map[String, Tensor[_]] = {
    val moduleConsts = model.findModules("Const")
      .map(_.asInstanceOf[Const[T, _]])
      .map(v => (v, v.value.shallowClone()))
    moduleConsts.foreach(_._1.value.set())
    val result = moduleConsts.map(v => (v._1.getName(), v._2)).toMap[String, Tensor[_]]
    require(result.size == moduleConsts.length, s"${model}'s Const node's name is duplicated," +
      s"please check your model.")
    result
  }

  private[bigdl] def putConsts[T: ClassTag](
        model: Container[_, _, T],
        consts: Map[String, Tensor[_]])(implicit ev: TensorNumeric[T]) : Unit = {
    val moduleConsts = model.findModules("Const")
      .map(_.asInstanceOf[Const[T, _]])
    moduleConsts.foreach{const =>
      val constValue = const.value.asInstanceOf[NumericWildcard]
      val constName = const.getName()
      constValue.asInstanceOf[Tensor[NumericWildcard]]
        .set(consts(constName).asInstanceOf[Tensor[NumericWildcard]])
    }
  }

  private def clearTensor[T: ClassTag](tensors: Array[Tensor[T]])
    (implicit ev: TensorNumeric[T]): Unit = {
    var i = 0
    while (i < tensors.length) {
      if (tensors(i) != null) {
        if (tensors(i).getTensorType == QuantizedType) {
          tensors(i).toQuantizedTensor.release()
        }

        tensors(i).set()
      }
      i += 1
    }
  }

  private[bigdl] def putWeightBias[T: ClassTag](
      broadcastWeightBias: Array[Tensor[T]],
      localModel: Module[T])(implicit ev: TensorNumeric[T]): Unit = {
    val localWeightBias = localModel.parameters()._1
    var i = 0
    while (i < localWeightBias.length) {
      if (localWeightBias(i) != null) {
        clearAndSet(localWeightBias(i), broadcastWeightBias(i))
      }
      i += 1
    }

    def clearAndSet(old: Tensor[T], other: Tensor[T]): Unit = {
      if (old.getTensorType == QuantizedType && other.getTensorType == QuantizedType) {
        val quantOld = old.asInstanceOf[QuantizedTensor[T]]
        val quantOther = other.asInstanceOf[QuantizedTensor[T]]

        if (quantOld.getNativeStorage != quantOther.getNativeStorage) {
          quantOld.release()
        }
      }

      old.set(other)
    }
  }

  private[bigdl] def initGradWeightBias[T: ClassTag](
      broadcastWeightBias: Array[Tensor[T]],
      localModel: Module[T])(implicit ev: TensorNumeric[T]): Unit = {
    val (localWeightBias, localGradWeightBias) = localModel.parameters()
    // init gradient with a compacted storage
    val storage = Storage[T](localGradWeightBias.map(_.nElement()).sum)
    val isQuantized = broadcastWeightBias.exists(_.getTensorType == QuantizedType)
    var i = 0
    while (i < localWeightBias.length) {
      if (localWeightBias(i) != null) {
        val wb = broadcastWeightBias(i)
        wb.getTensorType match {
          case QuantizedType =>
            localGradWeightBias(i).set(Tensor(1))
          case _ =>
            localGradWeightBias(i).set(storage, wb.storageOffset(), wb.size(), wb.stride())
        }
      }
      i += 1
    }
  }


  /**
   * This method is quite like [[org.apache.commons.lang3.SerializationUtils.deserialize]],
   * except `resolveClass` method of [[ObjectInputStream]] is overridden,
   * which fix potential [[ClassNotFoundException]] caused by uncertain `latestUserDefinedLoader`.
   */
  private[bigdl] def deserialize[T: ClassTag](objectData: Array[Byte]): T = {
    if (objectData == null) {
      throw new IllegalArgumentException("The byte[] must not be null")
    }
    deserialize[T](new ByteArrayInputStream(objectData))
  }

  /**
   * This method is quite like [[org.apache.commons.lang3.SerializationUtils.deserialize]],
   * except `resolveClass` method of [[ObjectInputStream]] is overridden,
   * which fix potential [[ClassNotFoundException]] caused by uncertain `latestUserDefinedLoader`.
   */
  private[bigdl] def deserialize[T: ClassTag](inputStream: InputStream): T = {
    if (inputStream == null) {
      throw new IllegalArgumentException("The InputStream must not be null")
    }
    var in: ObjectInputStream = null
    try {
      // stream closed in the finally
      in = new ObjectInputStream(inputStream) {
        override def resolveClass(desc: ObjectStreamClass): Class[_] = {
          Try(Class.forName(desc.getName, false, getClass.getClassLoader)
          ).getOrElse(super.resolveClass(desc))
        }
      }
      in.readObject().asInstanceOf[T]
    } catch {
      case ex: ClassCastException => throw new SerializationException(ex)
      case ex: ClassNotFoundException => throw new SerializationException(ex)
      case ex: IOException => throw new SerializationException(ex)
    } finally {
      if (in != null) Try(in.close())
    }
  }

}
