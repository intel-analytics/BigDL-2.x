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

package com.intel.analytics.zoo.persistent.memory

import com.intel.analytics.zoo.feature.common.persistent.memory.MemoryAllocator

sealed trait MemoryType

case object OptaneDC extends MemoryType

case object DRAM extends MemoryType

/**
 *
 * @param totalBytes
 */
abstract class NativeArray[T](totalBytes: Long, memoryType: MemoryType) {

  val memoryAllocator = MemoryAllocator.getInstance(memoryType)

  val startAddr: Long = memoryAllocator.allocate(totalBytes)

  assert(startAddr > 0, "Not enough memory!")
  assert(totalBytes > 0, "The size of bytes should be larger than 0!")

  val lastOffSet = startAddr + totalBytes

  var deleted: Boolean = false

  def get(i: Int): T

  def set(i: Int, value: T): Unit

  def free(): Unit = {
    if (!deleted) {
      memoryAllocator.free(startAddr)
      deleted = true
    }
  }

  protected def indexOf(i: Int): Long
}


