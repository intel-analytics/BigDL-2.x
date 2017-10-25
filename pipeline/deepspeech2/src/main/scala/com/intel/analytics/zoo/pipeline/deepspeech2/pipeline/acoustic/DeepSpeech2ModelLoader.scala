package com.intel.analytics.zoo.pipeline.deepspeech2.pipeline.acoustic

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

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn._
import org.apache.hadoop.fs.Path
import org.apache.log4j.Logger

import scala.language.existentials


object DeepSpeech2ModelLoader {

  val logger = Logger.getLogger(getClass)

  def loadModel(path: String): Module[Float] = {
    Module.load[Float](new Path(path).toString)
  }
}
