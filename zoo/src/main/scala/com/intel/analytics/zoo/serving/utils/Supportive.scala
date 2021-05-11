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

import org.apache.log4j.Logger

trait Supportive {
  def timing[T](name: String)(f: => T): T = {
    val begin = System.nanoTime()
    val result = f
    val end = System.nanoTime()
    val cost = (end - begin)
    Logger.getLogger(getClass).info(s"$name time elapsed [ ${cost / 1e6} ms ].")
    result
  }
}
