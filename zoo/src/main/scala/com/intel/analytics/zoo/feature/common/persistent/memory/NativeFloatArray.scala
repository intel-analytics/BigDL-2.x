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

package com.intel.analytics.zoo.feature.common.persistent.memory

import com.intel.analytics.zoo.persistent.memory.{MemoryType, NativeArray, OptaneDC}
import org.apache.spark.unsafe.Platform

object OptaneDCFloatArray {
  def apply(iterator: Iterator[Float], numOfRecord: Int): NativeFloatArray = {
    val aepArray = new NativeFloatArray(numOfRecord)
    var i = 0
    while (iterator.hasNext) {
      aepArray.set(i, iterator.next())
      i += 1
    }
    aepArray
  }
}

/**
 * An float array with fixed size stored in native memory.
 * @param recordNum number of item for this array.
 */
class NativeFloatArray(val recordNum: Int,
    sizeOfRecordByBytes: Int = 4,
    memoryType: MemoryType = OptaneDC) extends NativeArray[Float](
  recordNum * sizeOfRecordByBytes, memoryType) {

  override def get(i: Int): Float = {
    assert(!deleted)
    Platform.getFloat(null, indexOf(i))
  }

  def set(i: Int, value: Float): Unit = {
    assert(!deleted)
    Platform.putFloat(null, indexOf(i), value)
  }

  protected def indexOf(i: Int): Long = {
    val index = startAddr + i * sizeOfRecordByBytes
    assert(index <= lastOffSet)
    index
  }
}

object NativeVarLenFloatsArray {
  // Backward compatible with Spark.6
  val FLOAT_ARRAY_OFFSET = {
    var unsafe: sun.misc.Unsafe = null
    var _UNSAFE: sun.misc.Unsafe = null
    try {
      val unsafeField = classOf[sun.misc.Unsafe].getDeclaredField("theUnsafe")
      unsafeField.setAccessible(true)
      unsafe = unsafeField.get(null).asInstanceOf[(sun.misc.Unsafe)]
    } catch {
      case cause: Throwable =>
        unsafe = null
    }
    _UNSAFE = unsafe

    if (_UNSAFE != null) {
      _UNSAFE.arrayBaseOffset(classOf[Array[Float]])
    } else {
      0
    }
  }
}


class NativeVarLenFloatsArray(recordNum: Int, totalSizeByBytes: Long,
    memoryType: MemoryType = OptaneDC) extends NativeVarLenArray[Float](recordNum,
  totalSizeByBytes, memoryType, 4) {

  override def putSingle(offset: Long, s: Float): Unit = {
    Platform.putFloat(null, offset, s.asInstanceOf[Float])
  }

  override def get(i: Int): Array[Float] = {
    assert(isValidIndex(i))
    val recordLen = indexer(i)._2 / typeLen
    val result = new Array[Float](recordLen)
    Platform.copyMemory(null, indexOf(i), result,
      NativeVarLenFloatsArray.FLOAT_ARRAY_OFFSET, indexer(i)._2)
    return result
  }
}
