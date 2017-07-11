/*
 * Copyright 2016 The BigDL Authors.
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

package com.intel.analytics.bigdl.utils

import java.io._
import java.nio.file.{Files, Paths}

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FSDataInputStream, FSDataOutputStream, FileSystem, Path}
import org.apache.hadoop.io.IOUtils

object File {
  private[bigdl] val hdfsPrefix: String = "hdfs:"
  private[bigdl] val s3aPrefix: String = "s3a:"

  /**
   * Load torch object from a torch binary file
   *
   * @param fileName file name.
   * @return An instance of T
   */
  def loadTorch[T](fileName: String): T = {
    TorchFile.load[T](fileName)
  }

  /**
   * Save scala object into a torch binary file
   *
   * @param source  The object to be saved.
   * @param fileName file name to saving.
   * @param objectType The object type.
   * @param overWrite If over write.
   */
  def saveTorch(
      source: Any,
      fileName: String,
      objectType: TorchObject,
      overWrite: Boolean = false): Unit = {
    TorchFile.save(source, fileName, objectType, overWrite)
  }

  /**
   * Save scala object into a local/hdfs/s3 path
   *
   * Notice: S3 path should be like s3a://bucket/xxx.
   *
   * See (hadoop aws)[http://hadoop.apache.org/docs/r2.7.3/hadoop-aws/tools/hadoop-aws/index.html]
   * for details, if you want to save model to s3.
   *
   * @param obj object to be saved.
   * @param fileName local/hdfs output path.
   * @param isOverwrite if overwrite.
   */
  def save(obj: Serializable, fileName: String, isOverwrite: Boolean = false): Unit = {
    val conf = getConfiguration(fileName)
    save(obj, fileName, isOverwrite, conf)
  }

  private[bigdl] def getFileSystem(fileName: String): org.apache.hadoop.fs.FileSystem = {
    val src = new Path(fileName)
    val fs = src.getFileSystem(File.getConfiguration(fileName))
    require(fs.exists(src), src + " does not exists")
    fs
  }

  private[bigdl] def getConfiguration(fileName: String): Configuration = {
    if (fileName.startsWith(File.hdfsPrefix) || fileName.startsWith(s3aPrefix)) {
      new Configuration()
    } else {
      new Configuration(false)
    }
  }

  private def save(obj: Serializable, fileName: String, overwrite: Boolean,
                   conf: Configuration): Unit = {
    val dest = new Path(fileName)
    var fs: FileSystem = null
    var out: FSDataOutputStream = null
    var objFile: ObjectOutputStream = null
    try {
      fs = dest.getFileSystem(conf)
      if (fs.exists(dest)) {
        if (overwrite) {
          fs.delete(dest, true)
        } else {
          throw new RuntimeException(s"file $fileName already exists")
        }
      }
      out = fs.create(dest)
      val byteArrayOut = new ByteArrayOutputStream()
      objFile = new ObjectOutputStream(byteArrayOut)
      objFile.writeObject(obj)
      IOUtils.copyBytes(new ByteArrayInputStream(byteArrayOut.toByteArray), out, 1024, true)
    } finally {
      if (null != objFile) objFile.close()
      if (null != out) out.close()
      if (null != fs) fs.close()
    }
  }

  /**
   * Write file to HDFS.
   * @param obj
   * @param fileName
   * @param overwrite
   */
  def saveToHdfs(obj: Serializable, fileName: String, overwrite: Boolean): Unit = {
    require(fileName.startsWith(File.hdfsPrefix),
      s"hdfs path ${fileName} should have prefix 'hdfs:'")
    val dest = new Path(fileName)
    var fs: FileSystem = null
    var out: FSDataOutputStream = null
    var objFile: ObjectOutputStream = null
    try {
      fs = dest.getFileSystem(new Configuration())
      if (fs.exists(dest)) {
        if (overwrite) {
          fs.delete(dest, true)
        } else {
          throw new RuntimeException(s"file $fileName already exists")
        }
      }
      out = fs.create(dest)
      val byteArrayOut = new ByteArrayOutputStream()
      objFile = new ObjectOutputStream(byteArrayOut)
      objFile.writeObject(obj)
      IOUtils.copyBytes(new ByteArrayInputStream(byteArrayOut.toByteArray), out, 1024, true)
    } finally {
      if (null != objFile) objFile.close()
      if (null != out) out.close()
      if (null != fs) fs.close()
    }
  }

  /**
   * Load file from HDFS
   *
   * @param fileName
   */
  def loadFromHdfs[T](fileName: String): T = {
    val byteArrayOut = readHdfsByte(fileName)
    var objFile: ObjectInputStream = null
    try {
      objFile = new ObjectInputStream(new ByteArrayInputStream(byteArrayOut))
      val result = objFile.readObject()
      objFile.close()
      result.asInstanceOf[T]
    } finally {
      if (null != objFile) objFile.close()
    }
  }

  /**
   * Load a scala object from a local/hdfs/s3 path.
   *
   * Notice: S3 path should be like s3a://bucket/xxx.
   *
   * See (hadoop aws)[http://hadoop.apache.org/docs/r2.7.3/hadoop-aws/tools/hadoop-aws/index.html]
   * for details, if you want to load model from s3.
   *
   * @param fileName file name.
   */
  def load[T](fileName: String): T = {
    val conf = getConfiguration(fileName)
    load[T](fileName, conf)
  }

  /**
   * Load a scala object from a local/hdfs/s3 path.
   *
   * @param fileName file name.
   * @param conf hadoop Configuration.
   */
  private def load[T](fileName: String, conf: Configuration): T = {
    val src: Path = new Path(fileName)
    var fs: FileSystem = null
    var in: FSDataInputStream = null
    var objFile: ObjectInputStream = null
    try {
      fs = src.getFileSystem(conf)
      in = fs.open(src)
      val byteArrayOut = new ByteArrayOutputStream()
      IOUtils.copyBytes(in, byteArrayOut, 1024, true)
      objFile = new ObjectInputStream(new ByteArrayInputStream(byteArrayOut.toByteArray))
      val result = objFile.readObject()
      result.asInstanceOf[T]
    } finally {
      if (null != in) in.close()
      if (null != fs) fs.close()
      if (null != objFile) objFile.close()
    }
  }

  /**
   * load binary file from HDFS
   * @param fileName
   * @return
   */
  def readHdfsByte(fileName: String): Array[Byte] = {
    val src: Path = new Path(fileName)
    var fs: FileSystem = null
    var in: FSDataInputStream = null
    try {
      fs = src.getFileSystem(new Configuration())
      in = fs.open(src)
      val byteArrayOut = new ByteArrayOutputStream()
      IOUtils.copyBytes(in, byteArrayOut, 1024, true)
      byteArrayOut.toByteArray
    } finally {
      if (null != in) in.close()
      if (null != fs) fs.close()
    }
  }
}
