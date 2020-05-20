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

package com.intel.analytics.zoo.serving.http

import com.fasterxml.jackson.databind._
import com.fasterxml.jackson.module.scala.{DefaultScalaModule, ScalaObjectMapper}

trait SerializeSuported {
  def serialize(src: Object): String
  def deSerialize[T](clazz: Class[T], dest: String): T
}

class JacksonJsonSerializer extends SerializeSuported {
  val mapper = new ObjectMapper() with ScalaObjectMapper
  mapper.registerModule(DefaultScalaModule)
  mapper.configure(MapperFeature.ACCEPT_CASE_INSENSITIVE_PROPERTIES, true)
  mapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false)
  mapper.configure(SerializationFeature.FAIL_ON_EMPTY_BEANS, false)
  mapper.configure(SerializationFeature.INDENT_OUTPUT, true)
  mapper

  override def serialize(src: Object): String = {
    mapper.writeValueAsString(src)
  }

  override def deSerialize[T](clazz: Class[T], dest: String): T = {
    mapper.readValue[T](dest, clazz)
  }
}

object JsonUtil {
  val jacksonJsonSerializer = new JacksonJsonSerializer()
  def fromJson[T](clazz: Class[T], dest: String): T = jacksonJsonSerializer.deSerialize(clazz, dest)
  def toJson(value: Object): String = jacksonJsonSerializer.serialize(value)
}
