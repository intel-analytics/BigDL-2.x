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

package com.intel.analytics.zoo.transform.vision.image

import org.apache.log4j.Logger
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

object ImageFrame {
  val logger = Logger.getLogger(getClass)

  /**
   * create LocalImageFrame
   * @param data array of ImageFeature
   */
  def array(data: Array[ImageFeature]): LocalImageFrame = {
    new LocalImageFrame(data)
  }

  /**
   * create DistributedImageFrame
   * @param data rdd of ImageFeature
   */
  def rdd(data: RDD[ImageFeature]): DistributedImageFrame = {
    new DistributedImageFrame(data)
  }
}

class LocalImageFrame(var array: Array[ImageFeature]) {
  def apply(transformer: FeatureTransformer): LocalImageFrame = {
    array = array.map(transformer.transform)
    this
  }

  def toDistributed(sc: SparkContext): DistributedImageFrame = {
    new DistributedImageFrame(sc.parallelize(array))
  }
}

class DistributedImageFrame(var rdd: RDD[ImageFeature]) {
  def apply(transformer: FeatureTransformer): DistributedImageFrame = {
    rdd = transformer(rdd)
    this
  }
}
