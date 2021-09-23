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

package com.intel.analytics.zoo.serving.preprocessing

object DataType extends Enumeration{
  val IMAGE = value("IMAGE", 0)
  val TENSOR = value("TENSOR", 1)
  val SPARSETENSOR = value("SPARSETENSOR", 2)

  class DataTypeEnumVal(name: String, val value: Int) extends Val(nextId, name)

  protected final def value(name: String, value: Int): DataTypeEnumVal =
    new DataTypeEnumVal(name, value)
}
