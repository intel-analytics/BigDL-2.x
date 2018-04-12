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

package com.intel.analytics.zoo.models.common

import scala.collection.JavaConverters._

import com.intel.analytics.bigdl.nn.{Container, Module}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.serializer.{ContainerSerializable, DeserializeContext, ModuleSerializer}

import scala.reflect.ClassTag

/**
 * The base class for models in Analytics Zoo.
 *
 * @tparam A Input data type.
 * @tparam B Output data type.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
abstract class ZooModel[A <: Activity: ClassTag, B <: Activity: ClassTag, T: ClassTag]
(implicit ev: TensorNumeric[T]) extends Container[A, B, T] {

  /**
   * Override this method to define a model.
   */
  protected def buildModel(): AbstractModule[A, B, T]

  modules += buildModel()

  /**
   * The defined model, either from buildModel() or loaded from file.
   */
  def model: AbstractModule[A, B, T] = {
    if (modules.isEmpty) {
      throw new RuntimeException("No model found")
    }
    require(modules.length == 1,
      s"There should be exactly one model but found ${modules.length} models")
    modules(0).asInstanceOf[AbstractModule[A, B, T]]
  }

  // scalastyle:off
  def model_=(value: AbstractModule[A, B, T]): Unit = {
    modules.clear()
    modules.append(value)
  }
  // scalastyle:on

  override def updateOutput(input: A): B = {
    output = model.updateOutput(input)
    output
  }

  override def updateGradInput(input: A, gradOutput: B): A = {
    gradInput = model.updateGradInput(input, gradOutput)
    gradInput
  }

  override def accGradParameters(input: A, gradOutput: B): Unit = {
    model.accGradParameters(input, gradOutput)
  }
}

/**
 * The trait to be extended by model object to define some common static methods.
 */
trait ZooModelHelper {
  /**
   * Load an existing model definition (with weights).
   *
   * @param path The path for the pre-defined model.
   *             Local file system, HDFS and Amazon S3 are supported.
   *             HDFS path should be like "hdfs://[host]:[port]/xxx".
   *             Amazon S3 path should be like "s3a://bucket/xxx".
   * @param weightPath The path for pre-trained weights if any. Default is null.
   * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
   * @return
   */
  def load[T: ClassTag](path: String,
                        weightPath: String = null)(implicit ev: TensorNumeric[T]):
      AbstractModule[Activity, Activity, T] = {
    ZooModel.load[T](path, weightPath)
  }
}

object ZooModel {
  // Currently need to register ZooModelSerializable for each subclass of ZooModel.
  // Modify this if BigDL supports registering the parent class only.
  ModuleSerializer
    .registerModule("com.intel.analytics.zoo.models.textclassification.TextClassifier",
      ZooModelSerializable)

  def load[T: ClassTag](path: String,
                        weightPath: String = null)(implicit ev: TensorNumeric[T]):
      AbstractModule[Activity, Activity, T] = {
    Module.loadModule[T](path, weightPath)
  }
}

object ZooModelSerializable extends ContainerSerializable {
  override def loadSubModules[T: ClassTag](context: DeserializeContext,
                                           module: AbstractModule[Activity, Activity, T])
      (implicit ev: TensorNumeric[T]): Unit = {
    val zooModel = module.asInstanceOf[ZooModel[Activity, Activity, T]]
    val subModules = context.bigdlModule.getSubModulesList.asScala
    subModules.foreach(module => {
      val subModuleData = ModuleSerializer.load(DeserializeContext(module,
        context.storages, context.storageType, _copyWeightAndBias))
      zooModel.model = subModuleData.module
    })
  }
}
