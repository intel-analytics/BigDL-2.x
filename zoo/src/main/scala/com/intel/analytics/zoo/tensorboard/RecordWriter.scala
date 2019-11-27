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

package com.intel.analytics.zoo.tensorboard

import java.io.{File, FileOutputStream}

import com.google.common.primitives.{Ints, Longs}
import netty.Crc32c
import org.apache.hadoop.fs.{FSDataOutputStream, FileSystem, Path}
import org.tensorflow.util.Event

/**
 * A writer to write event protobuf to file by tensorboard's format.
 * @param file Support local path and HDFS path
 */
private[zoo] class RecordWriter(file: Path, fs: FileSystem) {
  val outputStream = if (file.toString.startsWith("hdfs://")) {
    // FSDataOutputStream couldn't flush data to localFileSystem in time. So reading summaries
    // will throw exception.
    fs.create(file, true, 1024)
  } else {
    // Using FileOutputStream when write to local.
    new FileOutputStream(new File(file.toString))
  }
  val crc32 = new Crc32c()
  def write(event: Event): Unit = {
    val eventString = event.toByteArray
    val header = Longs.toByteArray(eventString.length.toLong).reverse
    outputStream.write(header)
    outputStream.write(Ints.toByteArray(Crc32.maskedCRC32(crc32, header).toInt).reverse)
    outputStream.write(eventString)
    outputStream.write(Ints.toByteArray(Crc32.maskedCRC32(crc32, eventString).toInt).reverse)
    if (outputStream.isInstanceOf[FSDataOutputStream]) {
      // Flush data to HDFS.
      outputStream.asInstanceOf[FSDataOutputStream].hflush()
    }
  }

  def close(): Unit = {
    outputStream.close()
  }
}

private[zoo] object Crc32 {

  def maskedCRC32(crc32c: Crc32c, data: Array[Byte], offset: Int, length: Int): Long = {
    crc32c.reset()
    crc32c.update(data, offset, length)
    val x = u32(crc32c.getValue)
    u32(((x >> 15) | u32(x << 17)) + 0xa282ead8)
  }

  def maskedCRC32(crc32c: Crc32c, data: Array[Byte]): Long = {
    maskedCRC32(crc32c, data, 0, data.length)
  }

  def maskedCRC32(data: Array[Byte]): Long = {
    val crc32c = new Crc32c()
    maskedCRC32(crc32c, data)
  }

  def maskedCRC32(data: Array[Byte], offset: Int, length: Int): Long = {
    val crc32c = new Crc32c()
    maskedCRC32(crc32c, data, offset, length)
  }


  def u32(x: Long): Long = {
    x & 0xffffffff
  }

}
