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

package com.intel.analytics.zoo.pipeline.inference

import com.intel.analytics.bigdl.dataset.Sample
import org.slf4j.LoggerFactory

import scala.collection.JavaConverters._
import com.intel.analytics.bigdl.tensor.Tensor
import java.util.{List => JList}
import java.lang.{Float => JFloat}

trait InferenceSupportive {

  val logger = LoggerFactory.getLogger(getClass)

  def timing[T](name: String)(f: => T): T = {
    val begin = System.currentTimeMillis
    val result = f
    val end = System.currentTimeMillis
    val cost = (end - begin)
    logger.info(s"$name time elapsed [${cost / 1000} s, ${cost % 1000} ms].")
    result
  }

  def transferInputToSample(input: JList[JFloat], inputShape: Array[Int]):
    Sample[Float] = {
    require(input.size() == inputShape.reduce(_ * _), "data size not fit shape")
    val inputData = input.asScala.toArray.map(_.asInstanceOf[Float])
    Sample(Tensor(data = inputData, shape = inputShape))
  }

}
