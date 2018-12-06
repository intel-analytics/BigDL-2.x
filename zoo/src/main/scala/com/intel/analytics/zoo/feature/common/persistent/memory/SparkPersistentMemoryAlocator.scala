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

import com.intel.analytics.zoo.persistent.memory._
import org.apache.spark.SparkEnv

object MemoryAllocator {

  def getInstance(memoryType: MemoryType = DRAM): BasicMemoryAllocator = {
    if (memoryType == OptaneDC) {
      println("Using persistent memory")
      SparkPersistentMemoryAlocator.nativeAllocator
    } else {
      println("Using main memory")
      DRAMBasicMemoryAllocator.instance
    }
  }
}

object SparkPersistentMemoryAlocator {
  // TODO: Passing the path as a parameter via sparkconf?
  val memPaths = List("/mnt/pmem0", "/mnt/pmem1")
  val memSizePerByte = 248 * 1024 * 1024 * 1024
  val pathIndex = executorID % memPaths.length
  println(s"Executor: ${executorID()} is using ${memPaths(pathIndex)}")
  lazy val nativeAllocator = {
    val instance = PersistentMemoryAllocator.getInstance()
    instance.initialize(memPaths(pathIndex), memSizePerByte)
    instance
  }

  private def executorID(): Int = {
    if (SparkEnv.get.executorId.equals("driver")) {
      1
    } else {
      SparkEnv.get.executorId.toInt
    }
  }

  def allocate(size: Long): Long = {
    nativeAllocator.allocate(size)
  }

  def free(address: Long): Unit = {
    nativeAllocator.free(address)
  }

  def copy(destAddress: Long, srcAddress: Long, size: Long): Unit = {
    nativeAllocator.copy(destAddress, srcAddress, size)
  }
}
