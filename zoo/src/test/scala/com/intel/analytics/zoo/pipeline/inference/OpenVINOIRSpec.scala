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

import java.io.File
import java.util
import java.util.Arrays

import com.google.common.io.Files
import com.intel.analytics.bigdl.tensor.Tensor
import org.codehaus.plexus.util.FileUtils
import org.scalatest.{BeforeAndAfterAll, FunSuite, Matchers}
import org.slf4j.LoggerFactory

import scala.io.Source
import scala.language.postfixOps
import sys.process._
import scala.util.Random


@OpenVinoTest
class OpenVINOIRSpec extends FunSuite with Matchers with BeforeAndAfterAll
  with InferenceSupportive {
  val s3Url = "https://s3-ap-southeast-1.amazonaws.com"
  val s3DataUrl = s"$s3Url" +
    s"/analytics-zoo-models/openvino/Tests_faster_rcnn_resnet101_coco_2018_01_28"
  val url_ov_resnet_tests_inputdata1 = s"$s3DataUrl/inputdata_1"
  val url_ov_resnet_tests_inputdata2 = s"$s3DataUrl/inputdata_2"

  val logger = LoggerFactory.getLogger(getClass)
  var tmpDir: File = _
  var resnetModel: OpenVINOModel = _
  val Batch = 4
  val resnetInferenceModel: InferenceModel = new InferenceModel(3)
  val resnetInputShape = Array(Batch, 3, 224, 224)
  var resnetModelFilePath: String =
    "/home/intel/Data/DLModels/openvino/2018_R5/resnet_v1_50_i8.xml"
  var resnetWeightFilePath: String =
    "/home/intel/Data/DLModels/openvino/2018_R5/resnet_v1_50_i8.bin"
  val resnetDeviceType = DeviceType.CPU
  var resnetInputdata1FilePath: String = _
  var resnetInputdata2FilePath: String = _

  override def beforeAll() {
    tmpDir = Files.createTempDir()
    //    val dir = "/home/intel/Data/DLDataset/imagenet/origin/val_bmp"
    val dir = new File(s"${tmpDir.getAbsolutePath}/OpenVINOIRSpec")
      .getCanonicalPath

    s"wget -P $dir $url_ov_resnet_tests_inputdata1" !;
    s"wget -P $dir $url_ov_resnet_tests_inputdata2" !;

    s"ls -alh $dir" !;

    resnetInputdata1FilePath = s"$dir/inputdata_1"
    resnetInputdata2FilePath = s"$dir/inputdata_2"
    resnetModel = InferenceModelFactory.loadOpenVINOModelForIR(
      resnetModelFilePath,
      resnetWeightFilePath,
      resnetDeviceType
    )
    resnetInferenceModel.doLoadOpenVINO(
      resnetModelFilePath,
      resnetWeightFilePath
    )
  }

  override def afterAll() {
    FileUtils.deleteDirectory(tmpDir)
    resnetModel.release()
  }


  /*test("openvino model should throw exception if load failed") {
    val thrown = intercept[InferenceRuntimeException] {
      InferenceModelFactory.loadOpenVINOModelForIR(
        resnetModelFilePath + "error",
        resnetWeightFilePath,
        resnetDeviceType
      )
    }
    assert(thrown.getMessage.contains("xml"))
  }*/

  test("openvino model should load successfully") {
    println(s"resnetModel from IR loaded as $resnetModel")
    resnetModel shouldNot be(null)
    println(s"resnetInferenceModel from IR loaded as $resnetInferenceModel")
    resnetInferenceModel shouldNot be(null)
  }


  test("OpenVinoModel should predict dummy JTensor correctly") {
    val arrayInputs = new util.ArrayList[util.List[JTensor]]()
    for (_ <- 1 to Batch) {

      val input = new JTensor(Seq.fill(3 * 224 * 224)(Random.nextFloat)
        .toArray[Float],
        resnetInputShape)
      arrayInputs.add(Arrays.asList({input}))
    }
    val inputs = arrayInputs.subList(0, Batch - 1)

    val results1 = resnetModel.predict(inputs)
    val results2 = resnetInferenceModel.doPredict(inputs)


    val threads2 = List.range(0, 5).map(i => {
      new Thread() {
        override def run(): Unit = {
          val results = resnetInferenceModel.doPredict(inputs)
        }
      }
    })
    threads2.foreach(_.start())
    threads2.foreach(_.join())
  }

  test("OpenVinoModel should predict dummy Tensor correctly") {

    val inputs = Tensor(Array(Batch, 3, 224, 224)).rand().addSingletonDimension()
    val results1 = resnetModel.predict(inputs)
    val results2 = resnetInferenceModel.doPredict(inputs)


    val threads2 = List.range(0, 5).map(i => {
      new Thread() {
        override def run(): Unit = {
          val results = resnetInferenceModel.doPredict(inputs)
        }
      }
    })
    threads2.foreach(_.start())
    threads2.foreach(_.join())
  }


  test("OpenVinoModel should predict Image JTensor correctly") {
    val indata1 = Source.fromFile(resnetInputdata1FilePath).getLines().map(_.toFloat).toArray
    val indata2 = Source.fromFile(resnetInputdata2FilePath).getLines().map(_.toFloat).toArray
    println(indata1.length, indata2.length, 4 * 3 * 224 * 224)
    val input1 = new JTensor(indata1, resnetInputShape)
    val input2 = new JTensor(indata2, resnetInputShape)
    val inputs = Arrays.asList(
      Arrays.asList({
        input1
      }),
      Arrays.asList({
        input1
      }),
      Arrays.asList({
        input2
      }),
      Arrays.asList({
        input2
      }))

    val results1 = resnetModel.predict(inputs)
    val results2 = resnetInferenceModel.doPredict(inputs)
    

    val threads2 = List.range(0, 5).map(_ => {
      new Thread() {
        override def run(): Unit = {
          val results = resnetInferenceModel.doPredict(inputs)
        }
      }
    })
    threads2.foreach(_.start())
    threads2.foreach(_.join())

  }

  def almostEqual(x: Float, y: Float, precision: Float): Boolean = {
    (x - y).abs <= precision match {
      case true => true
      case false => println(x, y); false
    }
  }

}
