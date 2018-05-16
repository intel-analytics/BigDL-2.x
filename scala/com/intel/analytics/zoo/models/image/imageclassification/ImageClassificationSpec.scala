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
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.feature.image.ImageSet
import com.intel.analytics.zoo.models.common.ZooModel
import com.intel.analytics.zoo.models.image.common.ImageConfigure
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest
import org.apache.commons.io.FileUtils
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.{FlatSpec, Matchers}

import sys.process._


class ImageClassificationSpec extends ZooSpecHelper {
  val resource = getClass.getClassLoader.getResource("imagenet/n04370456/")

  def predictLocal(url: String, publisher: String): Unit = {
    val tmpFile = Files.createTempDir()
    val dir = new File(tmpFile.toString + "/models")
    val dirName = dir.getCanonicalPath
    val modelFileName = url.split("/").last

    val cmd = s"wget -P $dirName $url"
    val cmd_result = cmd !

    System.setProperty("bigdl.localMode", "true")
    Engine.init

    val image = ImageSet.read(resource.getFile)
    val model = ImageClassifier.loadModel[Float](dirName + "/" + modelFileName)
    model.saveModel("./imc.model", overWrite = true)
    var result = null.asInstanceOf[ImageSet]
    var config = null.asInstanceOf[ImageConfigure[Float]]
    if (!publisher.equalsIgnoreCase("analytics-zoo")) {
      val name = model.getName()
      val splits = name.split(("_"))
      config = ImageClassificationConfig[Float](splits(1), splits(2), splits(3))
      result = model.predictImageSet(image, config)
    } else {
      result = model.predictImageSet(image)
    }
    val predicted = result.toLocal().array.map(_ (ImageFeature.predict).toString)

    val image2 = ImageSet.read(resource.getFile)
    val model2 = ImageClassifier.loadModel[Float]("./imc.model")
    var result2 = null.asInstanceOf[ImageSet]
    if (!publisher.equalsIgnoreCase("analytics-zoo")) {
      result2 = model.predictImageSet(image2, config)
    } else {
      result2 = model.predictImageSet(image2)
    }

    val predicted2 = result2.toLocal().array.map(_ (ImageFeature.predict).toString)

    assert(predicted.length == predicted2.length)
    require(predicted.head
      .equals(predicted2.head) == true)
    if (tmpFile.exists()) FileUtils.deleteDirectory(tmpFile)
    System.clearProperty("bigdl.localMode")

    "rm -rf ./imc.model" !!
  }

  def predict(url: String, publisher: String): Unit = {
    val tmpFile = Files.createTempDir()
    val dir = new File(tmpFile.toString + "/models")
    val dirName = dir.getCanonicalPath
    val modelFileName = url.split("/").last

    val cmd = s"wget -P $dirName $url"
    val cmd_result = cmd !

    val conf = Engine.createSparkConf().setMaster("local[1]").setAppName("predictor")
    val sc = NNContext.getNNContext(conf)

    val model = ImageClassifier.loadModel[Float](dirName + "/" + modelFileName)
    model.saveModel("./imc.model", overWrite = true)
    val image = ImageSet.read(resource.getFile, sc)
    var result = null.asInstanceOf[ImageSet]
    var config = null.asInstanceOf[ImageConfigure[Float]]
    if (!publisher.equalsIgnoreCase("analytics-zoo")) {
      val name = model.getName()
      val splits = name.split(("_"))
      config = ImageClassificationConfig[Float](splits(1), splits(2), splits(3))
      result = model.predictImageSet(image, config)
    } else {
      result = model.predictImageSet(image)
    }
    val predicted = result.toDistributed().rdd.collect()

    val model2 = ImageClassifier.loadModel[Float]("./imc.model")
    val image2 = ImageSet.read(resource.getFile, sc)
    var result2 = null.asInstanceOf[ImageSet]
    if (!publisher.equalsIgnoreCase("analytics-zoo"))
    {
      result2 = model.predictImageSet(image2, config)
    } else {
      result2 = model.predictImageSet(image2)
    }
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
      "imageclassification/imagenet/analytics-zoo_inception-v1-quantize_imagenet_0.1.0",
      "analytics-zoo")
  }

  "ImageClassifier" should "predict inception-v1-quantize" in {
    predict("https://s3-ap-southeast-1.amazonaws.com/analytics-zoo-models/imageclassification/" +
      "imagenet/analytics-zoo_inception-v1-quantize_imagenet_0.1.0", "analytics-zoo")
  }

  "ImageClassifier" should "predict bigdl inception-v1-quantize locally" in {
    predictLocal("https://s3-ap-southeast-1.amazonaws.com/bigdl-models/imageclassification/" +
      "imagenet/bigdl_inception-v1-quantize_imagenet_0.4.0.model", "bigdl")
  }

  "ImageClassifier" should "predict bigdl inception-v1-quantize" in {
    predict("https://s3-ap-southeast-1.amazonaws.com/bigdl-models/imageclassification/imagenet/" +
      "bigdl_inception-v1-quantize_imagenet_0.4.0.model", "bigdl")
  }
}

class ImageClassifierSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
//    val tmpFile = Files.createTempDir()
//    val dir = new File(tmpFile.toString + "/models")
//    val dirName = dir.getCanonicalPath
//    val url = "https://s3-ap-southeast-1.amazonaws.com/bigdl-models/imageclassification/" +
//      "imagenet/bigdl_inception-v1-quantize_imagenet_0.4.0.model"
//    val modelFileName = url.split("/").last
//    val cmd = s"wget -P $dirName $url"
//    val cmd_result = cmd !
//
//    val model = ImageClassifier.loadModel[Float](dirName + "/" + modelFileName)
//    val input = Tensor[Float](Array(3, 224, 224)).rand()
//    ZooSpecHelper.testZooModelLoadSave(model.asInstanceOf[ZooModel[Tensor[Float], Tensor[Float],
    // Float]], input, ImageClassifier.loadModel[Float])
  }
}
