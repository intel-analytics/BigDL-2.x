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
package org.apache.spark.ml

import java.io._
import java.util.Base64

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.feature.common.Preprocessing
import org.apache.commons.lang3.SerializationUtils
import org.apache.spark.SparkContext
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.param.{Param, ParamPair, ParamValidators, Params}
import org.apache.spark.ml.util.DefaultParamsReader.Metadata
import org.apache.spark.ml.util.{DefaultParamsReader, DefaultParamsWriter}
import org.json4s.jackson.JsonMethods.{compact, parse, render}
import org.json4s.{DefaultFormats, JInt, JObject, JString, JValue}

object DefaultParamsWriterWrapper {
  def saveMetadata(
                    instance: Params,
                    path: String,
                    sc: SparkContext,
                    extraMetadata: Option[JObject] = None,
                    paramMap: Option[JValue] = None): Unit = {
    DefaultParamsWriter.saveMetadata(instance, path, sc, extraMetadata, paramMap)
  }

  def loadMetadata(path: String, sc: SparkContext, expectedClassName: String = ""): Metadata = {
    DefaultParamsReader.loadMetadata(path, sc, expectedClassName)
  }

  def getAndSetParams(instance: Params, metadata: Metadata): Unit = {
    DefaultParamsReader.getAndSetParams(instance, metadata)
  }
}


class PreprocessingParam[IN <: Any, OUT <: Any](parent: Params, name: String, doc: String,
    isValid: Preprocessing[IN, OUT] => Boolean)
  extends Param[Preprocessing[IN, OUT]](parent, name, doc, isValid) {

  def this(parent: Params, name: String, doc: String) =
    this(parent, name, doc, (_: Any) => true)

  override def w(value: Preprocessing[IN, OUT]): ParamPair[Preprocessing[IN, OUT]] =
    super.w(value)

  override def jsonEncode(value: Preprocessing[IN, OUT]): String = {
    val bytes = SerializationUtils.serialize(value)
    bytes.mkString(",")
  }

  override def jsonDecode(str: String): Preprocessing[IN, OUT] = {
    val bytes = str.split(",").map(_.toByte)
    SerializationUtils.deserialize[Preprocessing[IN, OUT]](bytes)
  }
}
