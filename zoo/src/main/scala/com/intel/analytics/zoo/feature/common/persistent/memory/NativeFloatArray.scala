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
