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

package com.intel.analytics.zoo.serving

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.EngineRef
import com.intel.analytics.zoo.serving.ClusterServing.logger
import com.intel.analytics.zoo.serving.utils.ClusterServingHelper
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext

import scala.util.Random

object testBroadcast {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.zoo.feature.image").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.zoo").setLevel(Level.INFO)

  def main(args: Array[String]): Unit = {
    val helper = new ClusterServingHelper()
    helper.initContext()
    helper.initArgs(args)
    val coreNumber = EngineRef.getCoreNumber()
    val eType = EngineRef.getEngineType()
    logger.info("Engine Type is " + eType)
    logger.info("Core number is running at " + coreNumber.toString)
    val model = helper.loadInferenceModel()

    val tensor = Tensor[Float](3, 224, 224).apply1((_ => Random.nextFloat()))

    var tensorArray: Array[Tensor[Float]] = Array()
    for (i <- 0 until 8) {
      tensorArray :+ tensor
    }

    val tensorRDD = helper.sc.parallelize(tensorArray)
    println("partitions " + tensorRDD.partitions.size)

    tensorRDD.map(x => {
      val thisModel = model.value
      thisModel.doPredict(x)

    }).collect()
  }
}
