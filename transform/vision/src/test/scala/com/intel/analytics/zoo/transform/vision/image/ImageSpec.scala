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


import com.google.common.io.Files
import com.intel.analytics.bigdl.utils.Engine
import org.apache.commons.io.FileUtils
import org.apache.spark.sql.SparkSession
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class ImageSpec extends FlatSpec with Matchers with BeforeAndAfter {
  var spark: SparkSession = null
  val resource = getClass.getClassLoader.getResource("image/")
  before {
    val conf = Engine.createSparkConf().setAppName("ImageSpec").setMaster("local[2]")
    spark = SparkSession.builder().config(conf).getOrCreate()
  }

  after {
    if (null != spark) spark.stop()
  }

  "read LocalImageFrame" should "work properly" in {
    val local = Image.read(resource.getFile)
    local.array.length should be(1)
    assert(local.array(0).uri.endsWith("000025.jpg"))
    assert(local.array(0).bytes.length == 95959)
    local.array(0).getImage().shape() should be((375, 500, 3))
  }

  "LocalImageFrame toDistributed" should "work properly" in {
    val local = Image.read(resource.getFile)
    local.array.foreach(x => println(x.uri, x.bytes.length))
    val imageFeature = local.toDistributed(spark.sparkContext).rdd.first()
    assert(imageFeature.uri.endsWith("000025.jpg"))
    assert(imageFeature.bytes.length == 95959)
    imageFeature.getImage().shape() should be((375, 500, 3))
  }

  "read DistributedImageFrame" should "work properly" in {
    val distributed = Image.read(resource.getFile, spark.sparkContext)
    val imageFeature = distributed.rdd.first()
    assert(imageFeature.uri.endsWith("000025.jpg"))
    assert(imageFeature.bytes.length == 95959)
    imageFeature.getImage().shape() should be((375, 500, 3))
  }

  "SequenceFile write and read" should "work properly" in {
    val tmpFile = Files.createTempDir()
    val dir = tmpFile.toString + "/parque"
    Image.writeSequenceFile(resource.getFile, dir, spark)

    val distributed = Image.readSequenceFile(dir, spark)
    val imageFeature = distributed.rdd.first()
    assert(imageFeature.uri.endsWith("000025.jpg"))
    assert(imageFeature.bytes.length == 95959)
    FileUtils.deleteDirectory(tmpFile)
  }
}
