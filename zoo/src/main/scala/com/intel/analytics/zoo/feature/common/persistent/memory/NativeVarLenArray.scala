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

import com.intel.analytics.zoo.persistent.memory.{MemoryType, NativeArray}

import scala.reflect.ClassTag

private[zoo] abstract class NativeVarLenArray[T: ClassTag](val recordNum: Int, totalSizeByBytes:
Long,
    memoryType: MemoryType, protected val typeLen: Int) extends
  NativeArray[Array[T]](totalSizeByBytes, memoryType) {

  // TODO: maybe this can be changed to native long array
  val indexer = new Array[(Long, Int)](recordNum)

  protected def isValidIndex(i: Int): Boolean = {
    indexer(i) != null
  }

  // TODO: would be slow if we put item one by one.
  def set(i: Int, bytes: Array[T]): Unit = {
    assert(!deleted)
    if (!isValidIndex(i)) {
      val startOffset = if (i == 0) {
        this.startAddr
      } else {
        assert(isValidIndex(i - 1))
        indexer(i - 1)._1 + indexer(i - 1)._2
      }
      indexer(i) = (startOffset, bytes.length * typeLen) // NB!!!
    }
    val startOffset = indexOf(i)
    var j = 0
    while(j < bytes.length) {
      putSingle(startOffset + j * typeLen, bytes(j))
      j += 1
    }
  }

  def putSingle(offset: Long, s: T): Unit

  def indexOf(i: Int): Long = {
    assert(isValidIndex(i))
    return indexer(i)._1
  }
}
