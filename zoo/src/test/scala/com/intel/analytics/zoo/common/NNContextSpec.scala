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

package com.intel.analytics.zoo.common

import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import org.apache.spark.SparkConf

class NNContextSpec extends ZooSpecHelper {

  "initNNContext" should "contain conf in spark-analytics-zoo.conf" in {
    val conf = new SparkConf()
      .setMaster("local[4]")
    val sc = NNContext.initNNContext(conf, "NNContext Test")
    sc.getConf.get("spark.serializer") should be
    ("org.apache.spark.serializer.JavaSerializer")
  }

}
