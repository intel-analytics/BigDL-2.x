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
import java.nio.file.attribute.PosixFilePermissions
import java.nio.file.{Path => JPath}

import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.Tensor
import org.apache.commons.io.filefilter.WildcardFileFilter
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FSDataInputStream, FSDataOutputStream, FileSystem, Path}
import org.apache.hadoop.io.IOUtils
import org.apache.log4j.Logger
import org.apache.spark.rdd.RDD

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

private[zoo] object Utils {

  private val logger = Logger.getLogger(getClass)

  @inline
  def timeIt[T](name: String)(f: => T): T = {
    val begin = System.nanoTime()
    val result = f
    val end = System.nanoTime()
    val cost = end - begin
    logger.debug(s"$name time [${cost / 1.0e9} s].")
    result
  }

  def activity2VectorBuilder(data: Activity):
  mutable.Builder[Tensor[_], Vector[Tensor[_]]] = {
    val vec = Vector.newBuilder[Tensor[_]]
    if (data.isTensor) {
      vec += data.asInstanceOf[Tensor[_]]
    } else {
      var i = 0
      while (i < data.toTable.length()) {
        vec += data.toTable(i + 1)
        i += 1
      }
    }
    vec
  }

  def listLocalFiles(path: String): Array[File] = {
    val files = new ArrayBuffer[File]()
    listFiles(path, files)
    files.toArray
  }

  /**
   * List files in local file system
   * @param path String
   * @param files File handles will be appended to files
   */
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
   * List paths of local or remote files (HDFS, S3 and FTP etc)
   * with FileSystem API
   * @param path String path
   * @param recursive Recursive or not
   * @return Array[String]
   */
  def listPaths(path: String, recursive: Boolean = false): Array[String] = {
    val fs = getFileSystem(path)
    // List remote or local files
    val res = new ArrayBuffer[String]()
    try {
      val files = fs.listFiles(new Path(path), recursive)
      while (files.hasNext) {
        val file = files.next()
        // Ignore dir
        if (!file.isDirectory) {
          // Add file paths (string) into ArrayBuffer
          res.append(file.getPath.toString)
        }
      }
    } catch {
      case _: FileNotFoundException => logger.warn(s"$path doesn't exist!")
      case _: IOException => logger.error(s"List paths of $path error!")
    }
    fs.close()
    res.toArray
  }

  /**
   * Read all bytes of file (local or remote) and return bytes Array.
   * WARNING: Don't use it to read large files. It may cause performance issue
   * and OOM.
   * @param path String
   * @return Array[Byte]
   */
  def readBytes(path: String): Array[Byte] = {
    var fs: FileSystem = null
    var in: FSDataInputStream = null
    try {
      fs = getFileSystem(path)
      in = fs.open(new Path(path))
      val byteArrayOut = new ByteArrayOutputStream()
      IOUtils.copyBytes(in, byteArrayOut, 1024, true)
      byteArrayOut.toByteArray
    } finally {
      if (null != in) in.close()
      if (null != fs) fs.close()
    }
  }

  /**
   * Read all bytes of multiple files (local or remote) and
   * return 2 dim bytes Array.
   * WARNING: Don't use it to read large files. It may cause performance issue
   * and OOM.
   * @param paths String paths in Array
   * @return 2 dim Byte Array
   */
  def readBytes(paths: Array[String]): Array[Array[Byte]] = {
    paths.map(readBytes)
  }

  /**
   * Write string lines into given path (local or remote file system)
   * @param path String path
   * @param lines String content
   */
  def writeLines(path: String, lines: String): Unit = {
    val fs = getFileSystem(path)
    val outStream = fs.create(new Path(path), true)
    try {
      outStream.writeBytes(lines)
    } finally {
      outStream.close()
    }
    fs.close()
  }

  /**
   * Get FileSystem (local or remote) from given file path
   * @param fileName file path (string)
   * @param newInstance always new instance if is set to true, otherwise will
   *                    get from cache (may shared with our connections)
   * @return hadoop.fs.FileSystem
   */
  def getFileSystem(fileName: String, newInstance: Boolean = true): FileSystem = {
    if (newInstance) {
      FileSystem.newInstance(new Path(fileName).toUri, new Configuration())
    } else {
      FileSystem.get(new Path(fileName).toUri, new Configuration())
    }
  }

  /**
   * Get FileSystem (local or remote) from given file path
   * @param fileName file path (hadoop.fs.Path)
   * @param newInstance always new instance if is set to true, otherwise will
   *                    get from cache (may shared with our connections)
   * @return hadoop.fs.FileSystem
   */
  def getFileSystem(fileName: Path, newInstance: Boolean): FileSystem = {
    if (newInstance) {
      FileSystem.newInstance(fileName.toUri, new Configuration())
    } else {
      FileSystem.get(fileName.toUri, new Configuration())
    }
  }

  /**
   * Create file in FileSystem (local or remote)
   * @param path String path
   * @param overwrite overwrite exiting file or not
   * @return
   */
  def create(path: String, overwrite: Boolean = false): DataOutputStream = {
    getFileSystem(path).create(new Path(path), overwrite)
  }

  /**
   * Open file in FileSystem (local or remote)
   * @param path String path
   * @return DataInputStream
   */
  def open(path: String): DataInputStream = {
    getFileSystem(path).open(new Path(path))
  }

  def appendPrefix(localPath: String): String = {
    if (!localPath.startsWith("file://")) {
      if (!localPath.startsWith("/")) {
        throw new Exception("local path must be a absolute path")
      } else {
        "file://" + localPath
      }
    } else {
      localPath
    }
  }

  def putLocalFileToRemote(localPath: String, remotePath: String,
                           isOverwrite: Boolean = false): Unit = {

    val path = appendPrefix(localPath)
    val inputStream = getFileSystem(path).open(new Path(path))
    saveStream(inputStream, fileName = remotePath, isOverwrite = isOverwrite)
  }

  def getRemoteFileToLocal(remotePath: String, localPath: String,
                           isOverwrite: Boolean = false): Unit = {
    val path = appendPrefix(localPath)
    val inputStream = getFileSystem(remotePath).open(new Path(remotePath))
    saveStream(inputStream, fileName = path, isOverwrite = isOverwrite)
  }

  def exists(path: String): Boolean = {
    val updatedPath = if (path.startsWith("/")) {
      "file://" + path
    } else {
      path
    }
    val fs = getFileSystem(updatedPath)
    var result = false
    try {
      result = fs.exists(new Path(updatedPath))
    } catch {
      case _: IOException => logger.error(s"Check existence of $path error!")
    } finally {
      fs.close()
    }
    result
  }

  def mkdirs(path: String): Unit = {
    val updatedPath = if (path.startsWith("/")) {
      "file://" + path
    } else {
      path
    }
    val fs = getFileSystem(updatedPath)
    try {
      fs.mkdirs(new Path(updatedPath))
    } catch {
      case _: IOException => logger.error(s"make directory of $path error!")
    } finally {
      fs.close()
    }
  }

  /**
   * Save bytes into given path (local or remote file system).
   * WARNING: Don't use it to read large files. It may cause performance issue
   * and OOM.
   * @param bytes bytes
   * @param fileName String path
   * @param isOverwrite Overwrite exiting file or not
   */
  def saveBytes(bytes: Array[Byte], fileName: String, isOverwrite: Boolean = false): Unit = {
    val stream = new ByteArrayInputStream(bytes)
    saveStream(stream, fileName, isOverwrite)
  }


  private def saveStream(stream: InputStream, fileName: String,
                              isOverwrite: Boolean = false): Unit = {
    var fs: FileSystem = null
    var out: FSDataOutputStream = null
    try {
      fs = getFileSystem(fileName)
      out = fs.create(new Path(fileName), isOverwrite)
      IOUtils.copyBytes(stream, out, 1024, true)
    } finally {
      if (null != out) out.close()
      if (null != fs) fs.close()
    }
  }

  def logUsageErrorAndThrowException(errMessage: String, cause: Throwable = null): Unit = {
    logger.error(s"********************************Usage Error****************************\n"
      + errMessage)
    throw new AnalyticsZooException(errMessage, cause)
  }

  def createTmpDir(prefix: String = "Zoo", permissions: String = "rwx------"): JPath = {
    java.nio.file.Files.createTempDirectory(prefix,
      PosixFilePermissions.asFileAttribute(PosixFilePermissions.fromString(permissions)))
  }
}

class AnalyticsZooException(message: String, cause: Throwable)
  extends Exception(message, cause) {

  def this(message: String) = this(message, null)
}

case class RDDWrapper[T](value: RDD[T])

