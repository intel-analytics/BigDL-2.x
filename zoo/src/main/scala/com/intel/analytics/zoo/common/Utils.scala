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

import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{File, T, Table}
import com.intel.analytics.zoo.common.NNContext.getClass
import org.apache.commons.io.filefilter.WildcardFileFilter
import org.apache.log4j.Logger

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

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

  def saveBytes(bytes: Array[Byte], fileName: String, isOverwrite: Boolean = false) : Unit = {
    File.saveBytes(bytes, fileName, isOverwrite)
  }

  def logUsageErrorAndThrowException(errMessage: String, cause: Throwable = null): Unit = {
    logger.error(s"********************************Usage Error****************************\n"
      + errMessage)
    throw new AnalyticsZooException(errMessage, cause)
  }

  def cat[@specialized(Float, Double) T: ClassTag]
    (tensors: Array[Tensor[T]])(implicit ev: TensorNumeric[T]): Tensor[T] = {
    val sizes = tensors.head.size()
    val newTensor = Tensor[T](Array(tensors.size) ++ sizes)

    for (i <- 1 to tensors.size) {
      newTensor.select(1, i).copy(tensors(i - 1))
    }
    newTensor
  }

  def cat[@specialized(Float, Double) T: ClassTag]
  (tables: Array[Table])(implicit ev: TensorNumeric[T]): Table = {
    val size = tables.head.length()
    val res = T()
    for (i <- 1 to size) {
      res.insert(cat(tables.map(_[Tensor[T]](i))))
    }
    res
  }

  def cat[@specialized(Float, Double) T: ClassTag]
  (activities: Array[Activity])(implicit ev: TensorNumeric[T]): Activity = {
    if (activities.head.isTensor) cat(activities.map(_.toTensor))
    else cat(activities.map(_.toTable))
  }

  def split[@specialized(Float, Double) T: ClassTag]
  (tensor: Tensor[T])(implicit ev: TensorNumeric[T]): Array[Tensor[T]] = {
    tensor.split(1)
  }

  def split[@specialized(Float, Double) T: ClassTag]
  (activity: Activity)(implicit ev: TensorNumeric[T]): Array[Activity] = {
    if (activity.isTensor) split(activity.toTensor)
    else {
      val table = activity.toTable
      require(table.length() <  3, "only support split table with length < 3")
      val data1 = table[Tensor[T]](1).split(1)
      val data2 = table[Tensor[T]](2).split(1)
      data1.zip(data2).map(x => T(x._1, x._2))
    }
  }
}

class AnalyticsZooException(message: String, cause: Throwable)
  extends Exception(message, cause) {

  def this(message: String) = this(message, null)
}

