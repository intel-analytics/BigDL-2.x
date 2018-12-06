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

object OptaneDCVarBytesArray {
  def apply(iterator: Iterator[Array[Byte]],
      recordNumber: Int, recordBytes: Int): NativeBytesArray = {
    val nativeArray = new NativeBytesArray(recordNumber, recordBytes)
    var i = 0
    while(iterator.hasNext) {
      nativeArray.set(i, iterator.next())
      i += 1
    }
    nativeArray
  }
}

class NativeVarLenBytesArray(val recordNum: Int, totalSizeByBytes: Long,
    memoryType: MemoryType = OptaneDC) extends
  NativeArray[Array[Byte]](totalSizeByBytes, memoryType) {

  // TODO: maybe this can be changed to native long array
  val indexer = new Array[(Long, Int)](recordNum)

  private def isValidIndex(i: Int): Boolean = {
      indexer(i) != null
  }
  override def get(i: Int): Array[Byte] = {
    assert(isValidIndex(i))
    val recordLen = indexer(i)._2
    val index = indexer(i)._1
    val result = new Array[Byte](recordLen)
    Platform.copyMemory(null, indexOf(i), result, Platform.BYTE_ARRAY_OFFSET, recordLen)
    return result
  }

  // TODO: would be slow if we put byte one by one.
  def set(i: Int, bytes: Array[Byte]): Unit = {
    assert(!deleted)
    if (!isValidIndex(i)) {
      val startOffset = if (i == 0) {
        this.startAddr
      } else {
        assert(isValidIndex(i - 1))
        indexer(i - 1)._1 + indexer(i - 1)._2
      }
      indexer(i) = (startOffset, bytes.length)
    }
    val startOffset = indexOf(i)
    var j = 0
    while(j < bytes.length) {
      Platform.putByte(null, startOffset + j, bytes(j))
      j += 1
    }
  }

  def indexOf(i: Int): Long = {
    assert(isValidIndex(i))
    return indexer(i)._1
  }
}


class NativeBytesArray(val numOfRecord: Long, val sizeOfRecordByByte: Int,
    memoryType: MemoryType = OptaneDC) extends
  NativeArray[Array[Byte]](numOfRecord * sizeOfRecordByByte, memoryType) {

  override def get(i: Int): Array[Byte] = {
    val result = new Array[Byte](sizeOfRecordByByte)
    Platform.copyMemory(null, indexOf(i), result, Platform.BYTE_ARRAY_OFFSET, sizeOfRecordByByte)
    return result
  }

  // TODO: would be slow if we put byte one by one.
  def set(i: Int, bytes: Array[Byte]): Unit = {
    assert(!deleted)
    val startOffset = indexOf(i)
    var j = 0
    while(j < bytes.length) {
      Platform.putByte(null, startOffset + j, bytes(j))
      j += 1
    }
  }

  def indexOf(i: Int): Long = {
    val index = startAddr + (i * sizeOfRecordByByte)
    assert(index <= lastOffSet)
    index
  }
}

