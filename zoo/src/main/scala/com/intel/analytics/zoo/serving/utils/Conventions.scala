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

package com.intel.analytics.zoo.serving.utils

import org.apache.arrow.vector.types.FloatingPointPrecision
import org.apache.arrow.vector.types.pojo.ArrowType

object Conventions {
  val SERVING_STREAM_NAME = "serving_stream"
  val SERVING_MODEL_TMP_DIR = "cluster-serving-model"
  val SERVING_CONF_TMP_PATH = "cluster-serving-conf.yaml"
  val ARROW_INT = new ArrowType.Int(32, true)
  val ARROW_FLOAT = new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE)
  val ARROW_BINARY = new ArrowType.Binary()
  val ARROW_UTF8 = new ArrowType.Utf8

  val MODEL_SECURED_KEY = "model_secured"
  val MODEL_SECURED_SECRET = "secret"
  val MODEL_SECURED_SALT = "salt"
}
