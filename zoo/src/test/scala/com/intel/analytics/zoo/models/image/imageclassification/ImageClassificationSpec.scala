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

package com.intel.analytics.zoo.models.image.imageclassification

import java.io.File

import com.google.common.io.Files
import com.intel.analytics.bigdl.transform.vision.image.ImageFeature
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.feature.image.ImageSet

import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest
import org.apache.commons.io.FileUtils
import org.scalatest.{FlatSpec, Matchers}
import sys.process._


class ImageClassificationSpec extends FlatSpec with Matchers {
  val resource = getClass.getClassLoader.getResource("imagenet/n04370456/")

  def predictLocal(url: String): Unit = {
    val tmpFile = Files.createTempDir()
    val dir = new File(tmpFile.toString + "/models")
    val dirName = dir.getCanonicalPath
    val modelFileName = url.split("/").last
    import sys.process._
    val cmd = s"wget -P $dirName $url"
    val cmd_result = cmd!

    System.setProperty("bigdl.localMode", "true")
    Engine.init

    val image = ImageSet.read(resource.getFile)
    val model = ImageClassifier.loadModel[Float](dirName + "/" + modelFileName)
    model.saveModel("./imc.model", overWrite = true)
    val result = model.predictImageSet(image)
    val predicted = result.toLocal().array.map(_(ImageFeature.predict).toString)

    val image2 = ImageSet.read(resource.getFile)
    val model2 = ImageClassifier.loadModel[Float]("./imc.model")
    val result2 = model2.predictImageSet(image2)
    val predicted2 = result2.toLocal().array.map(_(ImageFeature.predict).toString)

    assert(predicted.length == predicted2.length)
    require(predicted.head
      .equals(predicted2.head) == true)
    if (tmpFile.exists()) FileUtils.deleteDirectory(tmpFile)
    System.clearProperty("bigdl.localMode")

    "rm -rf ./imc.model" !!
  }

  def predict(url: String): Unit = {
    val tmpFile = Files.createTempDir()
    val dir = new File(tmpFile.toString + "/models")
    val dirName = dir.getCanonicalPath
    val modelFileName = url.split("/").last
    import sys.process._
    val cmd = s"wget -P $dirName $url"
    val cmd_result = cmd!

    val conf = Engine.createSparkConf().setMaster("local[1]").setAppName("predictor")
    val sc = NNContext.getNNContext(conf)

    val model = ImageClassifier.loadModel[Float](dirName + "/" + modelFileName)
    model.saveModel("./imc.model", overWrite = true)
    val image = ImageSet.read(resource.getFile, sc)
    val result = model.predictImageSet(image)
    val predicted = result.toDistributed().rdd.collect()

    val model2 = ImageClassifier.loadModel[Float]("./imc.model")
    val image2 = ImageSet.read(resource.getFile, sc)
    val result2 = model.predictImageSet(image2)
    val predicted2 = result2.toDistributed().rdd.collect()

    assert(predicted.length == predicted2.length)
    require(predicted.head.predict(ImageFeature.predict)
      .equals(predicted2.head.predict(ImageFeature.predict)) == true)
    if (tmpFile.exists()) FileUtils.deleteDirectory(tmpFile)

    if (sc != null) {
      sc.stop()
    }

    "rm -rf ./imc.model" !!
  }

  "ImageClassifier" should "predict inception-v1-quantize locally" in {
    predictLocal("https://s3-ap-southeast-1.amazonaws.com/analytics-zoo-models/" +
      "imageclassification/imagenet/analytics-zoo_inception-v1-quantize_imagenet_0.1.0")

    "rm -rf ./ssd.model" !!
  }

  "ImageClassifier" should "predict inception-v1-quantize" in {
    predict("https://s3-ap-southeast-1.amazonaws.com/analytics-zoo-models/imageclassification/" +
      "imagenet/analytics-zoo_inception-v1-quantize_imagenet_0.1.0")
  }

  "ImageClassifier" should "predict bigdl inception-v1-quantize locally" in {
    predictLocal("https://s3-ap-southeast-1.amazonaws.com/bigdl-models/imageclassification/" +
      "imagenet/bigdl_inception-v1-quantize_imagenet_0.4.0.model")
  }

  "ImageClassifier" should "predict bigdl inception-v1-quantize" in {
    predict("https://s3-ap-southeast-1.amazonaws.com/bigdl-models/imageclassification/imagenet/" +
      "bigdl_inception-v1-quantize_imagenet_0.4.0.model")
  }
}

class ImageClassifierSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    // TODO: Add test for saveModel and extract load from the above unit test
  }
}
