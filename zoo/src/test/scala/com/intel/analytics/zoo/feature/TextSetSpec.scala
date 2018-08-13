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

package com.intel.analytics.zoo.feature

import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.feature.text._
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class TextSetSpec extends FlatSpec with Matchers with BeforeAndAfter {
  var sc : SparkContext = _

  before {
    val conf = new SparkConf().setAppName("Test TextFeature").setMaster("local[*]")
    sc = NNContext.initNNContext(conf)
  }

  after {
    if (sc != null) {
      sc.stop()
    }
  }

  "DistributedTextSet Transformation" should "work properly" in {
    val text = new TextFeature("Hello world, please annotate my text world my my my", Some(1.0f))
    val textset = TextSet.rdd(sc.parallelize(Seq(text)))
    val transformed = textset.tokenize().normalize().indexize().shapeSequence(len = 5).genSample()
    transformed.getWordIndex
  }
}
