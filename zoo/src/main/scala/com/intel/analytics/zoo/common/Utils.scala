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

package com.intel.analytics.zoo.common

import java.io._

import org.apache.commons.io.filefilter.WildcardFileFilter
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.log4j.Logger

import scala.collection.mutable.ArrayBuffer

private[zoo] object Utils {

  private val logger = Logger.getLogger(getClass)

  def listLocalFiles(path: String): Array[File] = {
    val files = new ArrayBuffer[File]()
    listFiles(path, files)
    files.toArray
  }

  def listFiles(path: String, files: ArrayBuffer[File]): Unit = {
    val file = new File(path)
    if (file.isDirectory) {
      file.listFiles().foreach(x => listFiles(x.getAbsolutePath, files))
    } else if (file.isFile) {
      files.append(file)
    } else {
      val filter = new WildcardFileFilter(file.getName)
      file.getParentFile.listFiles(new FilenameFilter {
        override def accept(dir: File, name: String): Boolean = filter.accept(dir, name)
      }).foreach(x => listFiles(x.getAbsolutePath, files))
    }
  }

  /**
   * List local or remote files (HDFS, S3, FTP etc) with FileSystem API
   * @param path String path
   * @param recursive Recursive or not
   * @return Array[String]
   */
  def listFiles(path: String, recursive: Boolean = false): Array[String] = {
    val fspath = new Path(path)
    val fs = FileSystem.get(fspath.toUri, new Configuration())
    // List remote or local files
    val files = fs.listFiles(fspath, recursive)
    // Add file paths (string) into ArrayBuffer
    val res = new ArrayBuffer[String]()
    while (files.hasNext) {
      val file = files.next()
      // Ignore dir
      if (!file.isDirectory) {
        res.append(file.getPath.toString)
      }
    }
    res.toArray
  }

  /**
   * Read bytes of local or remote file
   * @param path String
   * @return Array[Byte]
   */
  def readBytes(path: String): Array[Byte] = {
    val fspath = new Path(path)
    // Get FileSystem
    val fs = FileSystem.get(fspath.toUri, new Configuration())
    val inputStream = fs.open(fspath)
    val data = new Array[Byte](fs.getFileStatus(new Path(path))
      .getLen.toInt)
    try {
      // Read all file bytes
      inputStream.readFully(data)
    } finally {
      inputStream.close()
    }
    data
  }

  /**
   * Write string lines into given path (local or remote file system)
   * @param path String path
   * @param lines String content
   */
  def writeLines(path: String, lines: String): Unit = {
    val fspath = new Path(path)
    val fs = FileSystem.get(fspath.toUri, new Configuration())
    val outStream = fs.create(fspath, true)
    try {
      outStream.writeBytes(lines)
    } finally {
      outStream.close()
    }
  }

  /**
   * Save bytes into given path (local or remote file system)
   * @param bytes bytes
   * @param fileName String path
   * @param isOverwrite Overwrite exiting file or not
   */
  def saveBytes(bytes: Array[Byte], fileName: String, isOverwrite: Boolean = false): Unit = {
    val fspath = new Path(fileName)
    val fs = FileSystem.get(fspath.toUri, new Configuration())
    val outStream = fs.create(
      fspath,
      isOverwrite)
    try {
      outStream.write(bytes)
    } finally {
      outStream.close()
    }
  }

  def logUsageErrorAndThrowException(errMessage: String, cause: Throwable = null): Unit = {
    logger.error(s"********************************Usage Error****************************\n"
      + errMessage)
    throw new AnalyticsZooException(errMessage, cause)
  }
}

class AnalyticsZooException(message: String, cause: Throwable)
  extends Exception(message, cause) {

  def this(message: String) = this(message, null)
}

