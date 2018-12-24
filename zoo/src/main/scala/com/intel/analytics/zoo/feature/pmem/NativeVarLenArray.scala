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

package com.intel.analytics.zoo.feature.pmem

//import org.apache.spark.unsafe.Platform

import scala.reflect.ClassTag

private[zoo] abstract class NativeVarLenArray[T: ClassTag](val recordNum: Int,
    totalSizeByBytes: Long,
    memoryType: MemoryType, protected val typeLen: Int) extends
  NativeArray[Array[T]](totalSizeByBytes, memoryType) {

  // TODO: maybe this can be changed to native long array
  val indexer = new Array[Long](recordNum)

  var nextValidOffSet: Long = startAddr

  var nextValidRecordId: Int = 0

  protected def isValidIndex(i: Int): Boolean = {
    i < recordNum && indexer(i) != null
  }

  protected def getRecordLength(i: Int): Int = {
    val startOffSet = if (i == 0) {
      startAddr
    } else {
      indexOf(i)
    }
    val nextOffSetTmp = if (i == nextValidRecordId - 1) {
      nextValidOffSet
    } else if (i < nextValidRecordId - 1){
      indexOf(i + 1)
    } else {
      throw new IllegalArgumentException(
        s"Invalid index: ${i}, nextValidRecordId: ${nextValidRecordId}")
    }
    ((nextOffSetTmp - startOffSet) / typeLen).toInt
  }

//  protected def getTypeOffSet(): Int



  // TODO: would be slow if we put item one by one.
  def set(i: Int, ts: Array[T]): Unit = {
    assert(!deleted)
    val curOffSet = if (i == 0) {
        this.startAddr
    } else {
        assert(isValidIndex(i - 1))
        nextValidOffSet
      }
    indexer(i) = curOffSet

    var j = 0
    while(j < ts.length) {
      putSingle(curOffSet + j * typeLen, ts(j))
      j += 1
    }

    if (i == nextValidRecordId) {
      nextValidRecordId = i + 1
      nextValidOffSet = curOffSet + ts.length * typeLen
    }
  }

  def putSingle(offset: Long, s: T): Unit

  def indexOf(i: Int): Long = {
    assert(isValidIndex(i))
    return indexer(i)
  }
}
