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

package com.intel.analytics.zoo.models.image.objectdetection

import com.intel.analytics.bigdl.transform.vision.image.ImageFeature
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.feature.image.ImageSet
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest
import org.apache.spark.{SparkConf, SparkContext}
import org.opencv.imgcodecs.Imgcodecs

import scala.language.postfixOps
import sys.process._
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

/*

class ObjectDetectionSpec extends FlatSpec with Matchers with BeforeAndAfter {
  var sc : SparkContext = _

  before {
    val conf = new SparkConf().setAppName("Test ObjectDetector").setMaster("local[1]")
    sc = NNContext.initNNContext(conf)
    val s = s"wget -q -O ./ssd.model https://s3-ap-southeast-1.amazonaws.com/" +
      s"analytics-zoo-models/object-detection/analytics-zoo_ssd-mobilenet" +
      s"-300x300_PASCAL_0.1.0.model" !!
    val s2 = s"wget -q -O ./bigdl_ssd.model https://s3-ap-southeast-1.amazonaws.com/" +
      s"bigdl-models/object-detection/bigdl_ssd-mobilenet-300x300_PASCAL_0.4.0.model" !!
  }

  after {
    if (sc != null) {
      sc.stop()
    }
    "rm -rf ./ssd.model" !!

    "rm -rf ./bigdl_ssd.model" !!
  }

  "ObjectDetector model" should "be able to work" in {
    val resource = getClass.getClassLoader.getResource("pascal/")
    val data = ImageSet.read(resource.getFile, sc, 1)
    val model = ObjectDetector.loadModel[Float]("./ssd.model")
    val output = model.predictImageSet(data)

    model.saveModel("./ssd2.model", overWrite = true)
    val loadedModel = ObjectDetector.loadModel[Float]("./ssd2.model")
      .asInstanceOf[ObjectDetector[Float]]
    require(loadedModel.modules.length == 1)
    val data2 = ImageSet.read(resource.getFile, sc, 1)
    val output2 = loadedModel.predictImageSet(data2)
    val res = output.toDistributed().rdd.collect()
    val res2 = output2.toDistributed().rdd.collect()
    require(res.length == res2.length)
    require(res.head.predict(ImageFeature.predict)
      .equals(res2.head.predict(ImageFeature.predict)) == true)

    "rm -rf ./ssd2.model" !!
  }

  "ObjectDetector model" should "be able to work with bigdl model" in {
    val resource = getClass.getClassLoader.getResource("pascal/")
    val data = ImageSet.read(resource.getFile, sc, 1)
    val model = ObjectDetector.loadModel[Float]("./bigdl_ssd.model")
    val name = model.getName()
    val splits = name.split(("_"))
    val config = ObjectDetectionConfig[Float](splits(1), splits(2), splits(3))
    val output = model.predictImageSet(data, config)

    model.saveModel("./ssd2.model", overWrite = true)
    val loadedModel = ObjectDetector.loadModel[Float]("./ssd2.model")
      .asInstanceOf[ObjectDetector[Float]]
    require(loadedModel.modules.length == 1)
    val data2 = ImageSet.read(resource.getFile, sc, 1)
    val output2 = loadedModel.predictImageSet(data2, config)
    val res = output.toDistributed().rdd.collect()
    val res2 = output2.toDistributed().rdd.collect()
    require(res.length == res2.length)
    require(res.head.predict(ImageFeature.predict)
      .equals(res2.head.predict(ImageFeature.predict)) == true)

    "rm -rf ./ssd2.model" !!
  }

  "ObjectDetector model" should "be able to work with png" in {
    val resource = getClass.getClassLoader.getResource("png/")
    val data = ImageSet.read(resource.getFile, sc, 1, imageCodec = Imgcodecs.CV_LOAD_IMAGE_COLOR)
    val model = ObjectDetector.loadModel[Float]("./ssd.model")
    val output = model.predictImageSet(data)
    output.toDistributed().rdd.collect()

    "rm -rf ./ssd.model" !!
  }
}
*/
class ObjectDetectorSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    // TODO: Extract save and load from the above unit test
  }
}
