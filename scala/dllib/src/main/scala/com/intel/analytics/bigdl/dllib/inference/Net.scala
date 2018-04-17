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

package com.intel.analytics.zoo.pipeline.api

import java.nio.ByteOrder

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{File, Shape}
import com.intel.analytics.bigdl.utils.caffe.CaffeLoader
import com.intel.analytics.bigdl.utils.serializer.ModuleLoader
import com.intel.analytics.bigdl.utils.tf.{Session, TensorflowLoader}

import scala.reflect.ClassTag

/**
 * A placeholder to add layer's utilities
 */
trait Net {

  private def toClazz(obj: Object) = {
    obj match {
      case s: Shape => Class.forName("com.intel.analytics.bigdl.utils.Shape")
      case _ => obj.getClass()
    }
  }

  private[zoo] def callByName(methodName: String, args: Object*): Object = {
    val clazz = Class.forName("com.intel.analytics.bigdl.nn.keras.KerasLayer")
    val method = clazz.getMethod(methodName, args.map(toClazz(_)): _*)
    method.invoke(this, args: _*)
  }
}

object Net {
  /**
   * Load model from path.
   *
   * @param path path to save module, local file system, HDFS and Amazon S3 is supported.
   *             HDFS path should be like "hdfs://[host]:[port]/xxx"
   *             Amazon S3 path should be like "s3a://bucket/xxx"
   * @param weightPath : where weight is stored
   * @tparam T numeric type
   * @return model loaded from path
   */
  def loadModule[T: ClassTag](path : String,
      weightPath : String = null)(implicit ev: TensorNumeric[T])
  : AbstractModule[Activity, Activity, T] = {
    ModuleLoader.loadFromFile(path, weightPath)
  }

  def loadTorch[T: ClassTag](path : String) : AbstractModule[Activity, Activity, T] = {
    File.loadTorch[AbstractModule[Activity, Activity, T]](path)
  }

  /**
   * Loaf caffe trained model from prototxt and weight files
   * @param defPath  caffe model definition file path
   * @param modelPath caffe model binary file containing weight and bias
   */
  def loadCaffeModel[T: ClassTag](defPath: String, modelPath: String)(
      implicit ev: TensorNumeric[T]): AbstractModule[Activity, Activity, T] = {
    CaffeLoader.loadCaffe[T](defPath, modelPath)._1
      .asInstanceOf[AbstractModule[Activity, Activity, T]]
  }

  /**
   * Load tensorflow model from its saved protobuf file.
   * @param graphFile where is the protobuf model file
   * @param inputs input node names
   * @param outputs output node names, the output tensor order is same with the node order
   * @param byteOrder byte order in the tensorflow file. The default value is little endian
   * @param binFile where is the model variable file
   * @return BigDL model
   */
  def loadTF[T: ClassTag](graphFile: String, inputs: Seq[String], outputs: Seq[String],
      byteOrder: ByteOrder = ByteOrder.LITTLE_ENDIAN,
      binFile: Option[String] = None)(
      implicit ev: TensorNumeric[T]): AbstractModule[Activity, Activity, T] = {

    TensorflowLoader.load(graphFile, inputs, outputs, byteOrder, binFile)
      .asInstanceOf[AbstractModule[Activity, Activity, T]]
  }

  /**
   * Load tensorflow checkpoints
   * @param graphFile
   * @param binFile
   * @tparam T
   * @return
   */
  def tensorflowCheckpoints[T: ClassTag](graphFile: String, binFile: String,
      byteOrder: ByteOrder = ByteOrder.LITTLE_ENDIAN)(implicit ev: TensorNumeric[T]): Session[T] = {
    TensorflowLoader.checkpoints(graphFile, binFile, byteOrder)
  }
}
