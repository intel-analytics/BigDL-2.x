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

package com.intel.analytics.zoo.models.image.objectdetection.ssd

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Table
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.feature.image._
import com.intel.analytics.zoo.models.common.ZooModel
import com.intel.analytics.zoo.models.image.objectdetection.ObjectDetector
import com.intel.analytics.zoo.models.image.objectdetection.common.dataset.Imdb
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest
import org.apache.log4j.Logger
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.language.postfixOps

class SSDSpec extends FlatSpec with Matchers with BeforeAndAfter {
  var sc : SparkContext = _

  before {
    val conf = new SparkConf().setAppName("Test ObjectDetector").setMaster("local[1]")
    sc = NNContext.initNNContext(conf)
  }

  after {
    if (sc != null) {
      sc.stop()
    }
  }

  "SSDVGG model" should "be able to work" in {
    val model = SSDVGG[Float](20)
    val resource = getClass().getClassLoader().getResource("VOCdevkit")
    val voc = Imdb.getImdb("voc_2007_testcode", resource.getPath)
    val roidb = voc.getRoidb(true).array.toIterator
    val imgAug = ImageBytesToMat() ->
      ImageRoiNormalize() ->
      ImageColorJitter() ->
      ImageRandomPreprocessing(ImageExpand() -> ImageRoiProject(), 0.5) ->
      ImageRandomSampler() ->
      ImageResize(300, 300, -1) ->
      ImageRandomPreprocessing(ImageHFlip() -> ImageRoiHFlip(), 0.5) ->
      ImageChannelNormalize(123f, 117f, 104f) ->
      ImageMatToFloats(validHeight = 300, validWidth = 300) ->
      RoiImageToSSDBatch(2)
    val out = imgAug(roidb)
    val input = out.next()
    val output = model.forward(input.getInput)

  }

}

class SSDVGGSerialTest extends ModuleSerializationTest {
  protected val logger = Logger.getLogger(getClass)
  override def test(): Unit = {
    val model1 = SSDVGG[Float](20)

    val resource = getClass().getClassLoader().getResource("VOCdevkit")
    val voc = Imdb.getImdb("voc_2007_testcode", resource.getPath)
    val roidb = voc.getRoidb(true).array.toIterator
    val imgAug = ImageBytesToMat() ->
      ImageRoiNormalize() ->
      ImageColorJitter() ->
      ImageRandomPreprocessing(ImageExpand() -> ImageRoiProject(), 0.5) ->
      ImageRandomSampler() ->
      ImageResize(300, 300, -1) ->
      ImageRandomPreprocessing(ImageHFlip() -> ImageRoiHFlip(), 0.5) ->
      ImageChannelNormalize(123f, 117f, 104f) ->
      ImageMatToFloats(validHeight = 300, validWidth = 300) ->
      RoiImageToSSDBatch(2)
    val out = imgAug(roidb)
    val input = out.next()
    val serFile = java.io.File.createTempFile("UnitTest", "SSDSpecBase")
    logger.info(s"created file $serFile")
    model1.saveModel(serFile.getAbsolutePath, overWrite = true)
    val model2 = ZooModel.loadModel[Float](serFile.getAbsolutePath)
    val output1 = model1.forward(input.getInput)
    val output2 = model2.forward(input.getInput)
    output2.asInstanceOf[Table].keySet.
      sameElements(output1.asInstanceOf[Table].keySet) should be (true)
    output2.asInstanceOf[Table].equals(output1.asInstanceOf[Table]) should be (true)
    if (serFile.exists()) serFile.delete()
  }
}
