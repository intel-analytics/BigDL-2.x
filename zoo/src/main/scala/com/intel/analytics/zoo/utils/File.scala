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

package com.intel.analytics.zoo.utils

import java.io._

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}

object File {
  private[zoo] val hdfsPrefix: String = "hdfs:"
  private[zoo] val s3aPrefix: String = "s3a:"

  private[zoo] def getConfiguration(fileName: String): Configuration = {
    if (fileName.startsWith(File.hdfsPrefix) || fileName.startsWith(s3aPrefix)) {
      new Configuration()
    } else {
      new Configuration(false)
    }
  }
}

/**
 * FileReader.
 * @param fileName
 */
private[zoo] class FileReader(fileName: String) {
  private var inputStream: InputStream = null
  private val conf = File.getConfiguration(fileName)
  private val path = new Path(fileName)
  private val fs: FileSystem = path.getFileSystem(conf)

  /**
   * get an InputStream
   * @return
   */
  def open(): InputStream = {
    require(inputStream == null, s"File $fileName has been opened already.")
    require(fs.exists(path), s"$fileName is empty!")
    inputStream = fs.open(path)
    inputStream
  }

  /**
   * close the resources.
   */
  def close(): Unit = {
    if (null != inputStream) inputStream.close()
    fs.close()
  }
}

object FileReader {
  private[zoo] def apply(fileName: String): FileReader = {
    new FileReader(fileName)
  }
}

/**
 * FileWriter.
 * @param fileName
 */
private[zoo] class FileWriter(fileName: String) {
  private var outputStream: OutputStream = null
  private val conf = File.getConfiguration(fileName)
  private val path = new Path(fileName)
  private val fs: FileSystem = path.getFileSystem(conf)

  /**
   * get an OutputStream
   * @param overwrite if overwrite
   * @return
   */
  def create(overwrite: Boolean = false): OutputStream = {
    require(outputStream == null, s"File $fileName has been created already.")
    if (!overwrite) {
      require(!fs.exists(path), s"$fileName already exists!")
    }
    outputStream = fs.create(path, overwrite)
    outputStream
  }

  /**
   * close the resources.
   */
  def close(): Unit = {
    if (null != outputStream) outputStream.close()
    fs.close()
  }
}

object FileWriter {
  private[zoo] def apply(fileName: String): FileWriter = {
    new FileWriter(fileName)
  }
}


