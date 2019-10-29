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

package com.intel.analytics.zoo.pipeline.estimator

import com.intel.analytics.zoo.pipeline.inference.InferenceSupportive
import org.slf4j.LoggerFactory

trait EstimateSupportive extends InferenceSupportive {

  def throughputing[T](name: String, batch: Int)(f: => T): T = {
    val begin = System.currentTimeMillis
    val result = f
    val end = System.currentTimeMillis
    val cost = (end - begin)
    val throughput = batch.toDouble / cost * 1000
    EstimateSupportive.logger.info(
      s"$name time elapsed [${cost / 1000} s, ${cost % 1000} ms], " +
        s"throughput: ${throughput} records/second.")
    result
  }

  def throughputingWithLoss[T](name: String, batch: Int)(f: => T): T = {
    val begin = System.currentTimeMillis
    val result = f
    val end = System.currentTimeMillis
    val cost = (end - begin)
    val throughput = batch.toDouble / cost * 1000
    EstimateSupportive.logger.info(
      s"$name time elapsed [${cost / 1000} s, ${cost % 1000} ms], " +
        s"throughput: ${throughput} records/second, loss: ${result}.")
    result
  }

}

object EstimateSupportive {
  val logger = LoggerFactory.getLogger(getClass)
}
