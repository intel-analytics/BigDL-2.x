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
import java.util.{Arrays, Properties}

import com.google.common.io.Files
import org.scalatest._

import scala.io.Source
import scala.language.postfixOps
import sys.process._

@OpenVinoTest
class OpenVinoInferenceModelSuite extends FunSuite with Matchers with BeforeAndAfterAll
  with InferenceSupportive {

  var dataStoreUrl = "https://s3-ap-southeast-1.amazonaws.com"
  try {
    val prop = new Properties()
    prop.load(this.getClass.getResourceAsStream("/app.properties"))
    dataStoreUrl = prop.getProperty("data-store-url")
  } catch { case e: Exception =>
    dataStoreUrl = "https://s3-ap-southeast-1.amazonaws.com"
  }

  val url_ov = s"$dataStoreUrl/analytics-zoo-models/openvino"
  // val url_ov = "http://10.239.45.10:8081/repository/raw/openvinotests"
  val url_ov_fasterrcnn_ir = s"$url_ov/IR_faster_rcnn_resnet101_coco_2018_01_28"
  val url_ov_fasterrcnn_ir_bin = s"$url_ov_fasterrcnn_ir/frozen_inference_graph.bin"
  val url_ov_fasterrcnn_ir_xml = s"$url_ov_fasterrcnn_ir/frozen_inference_graph.xml"
  val url_ov_fasterrcnn_tests = s"$url_ov/Tests_faster_rcnn_resnet101_coco_2018_01_28"
  val url_ov_fasterrcnn_tests_inputdata1 = s"$url_ov_fasterrcnn_tests/inputdata_1"
  val url_ov_fasterrcnn_tests_inputdata2 = s"$url_ov_fasterrcnn_tests/inputdata_2"
  val url_ov_fasterrcnn_tests_outputdata1 = s"$url_ov_fasterrcnn_tests/outputdata_1"
  val url_ov_fasterrcnn_tests_outputdata2 = s"$url_ov_fasterrcnn_tests/outputdata_2"
  val url_ov_fasterrcnn_tf = s"$url_ov/TF_faster_rcnn_resnet101_coco_2018_01_28"
  val url_ov_fasterrcnn_tf_pb = s"$url_ov_fasterrcnn_tf/frozen_inference_graph.pb"
  val url_ov_fasterrcnn_tf_json = s"$url_ov_fasterrcnn_tf/faster_rcnn_support.json"
  val url_ov_fasterrcnn_tf_conf = s"$url_ov_fasterrcnn_tf/pipeline.config"

  var dir: File = _
  var fasterrcnnModel1: OpenVinoInferenceModel = _
  var fasterrcnnModel2: OpenVinoInferenceModel = _
  val fasterrcnnInferenceModel1: InferenceModel = new InferenceModel(3)
  val fasterrcnnInferenceModel2: InferenceModel = new InferenceModel(3)
  val fasterrcnnInputShape = Array(1, 3, 600, 600)
  var fasterrcnnInputdata1FilePath: String = _
  var fasterrcnnInputdata2FilePath: String = _
  var fasterrcnnOutputdata1FilePath: String = _
  var fasterrcnnOutputdata2FilePath: String = _

  override def beforeAll() {
    val tmpDir = Files.createTempDir()
    dir = new File(s"${tmpDir.getAbsolutePath}/OpenVinoInferenceModelSpec")
    val dirName: String = dir.getCanonicalPath

    s"wget -P $dirName $url_ov_fasterrcnn_ir_bin" !;
    s"wget -P $dirName $url_ov_fasterrcnn_ir_xml" !;
    s"wget -P $dirName $url_ov_fasterrcnn_tests_inputdata1" !;
    s"wget -P $dirName $url_ov_fasterrcnn_tests_inputdata2" !;
    s"wget -P $dirName $url_ov_fasterrcnn_tests_outputdata1" !;
    s"wget -P $dirName $url_ov_fasterrcnn_tests_outputdata2" !;
    s"wget -P $dirName $url_ov_fasterrcnn_tf_pb" !;
    s"wget -P $dirName $url_ov_fasterrcnn_tf_json" !;
    s"wget -P $dirName $url_ov_fasterrcnn_tf_conf" !;
    s"ls -alh $dirName" !;

    val fasterrcnnModelFilePath = s"$dirName/frozen_inference_graph.xml"
    val fasterrcnnWeightFilePath = s"$dirName/frozen_inference_graph.bin"
    val faserrcnnFrozenModelFilePath = s"$dirName/frozen_inference_graph.pb"
    val faserrcnnPipelineConfigFilePath = s"$dirName/pipeline.config"
    val faserrcnnExtensionsConfigFilePath = s"$dirName/faster_rcnn_support.json"
    val fasterrcnnDeviceType = DeviceType.CPU
    fasterrcnnInputdata1FilePath = s"$dirName/inputdata_1"
    fasterrcnnInputdata2FilePath = s"$dirName/inputdata_2"
    fasterrcnnOutputdata1FilePath = s"$dirName/outputdata_1"
    fasterrcnnOutputdata2FilePath = s"$dirName/outputdata_2"


    fasterrcnnModel1 = InferenceModelFactory.loadOpenvinoInferenceModelForIR(
      fasterrcnnModelFilePath, fasterrcnnWeightFilePath, fasterrcnnDeviceType)
    fasterrcnnModel2 = InferenceModelFactory.loadOpenvinoInferenceModelForTF(
      faserrcnnFrozenModelFilePath, faserrcnnPipelineConfigFilePath,
      faserrcnnExtensionsConfigFilePath, fasterrcnnDeviceType)
    fasterrcnnInferenceModel1.
      doLoadOpenvinoIR(fasterrcnnModelFilePath, fasterrcnnWeightFilePath, fasterrcnnDeviceType)
    fasterrcnnInferenceModel2.doLoadTensorflowModelAsOpenvino(
      faserrcnnFrozenModelFilePath, faserrcnnPipelineConfigFilePath,
      faserrcnnExtensionsConfigFilePath, fasterrcnnDeviceType)
  }

  override def afterAll() {
    dir.delete()
    fasterrcnnModel1.release()
    fasterrcnnModel2.release()
  }

  test("openvino model should load successfully") {
    println(s"fasterrcnnModel from openvino ir loaded as $fasterrcnnModel1")
    fasterrcnnModel1 shouldNot be(null)
    println(s"fasterrcnnModel from tensorflow pb loaded as $fasterrcnnModel2")
    fasterrcnnModel2 shouldNot be(null)
    println(s"fasterrcnnInferenceModel from openvino ir loaded as $fasterrcnnInferenceModel1")
    fasterrcnnInferenceModel1 shouldNot be(null)
    println(s"fasterrcnnInferenceModel from tensorflow pb loaded as $fasterrcnnInferenceModel2")
    fasterrcnnInferenceModel2 shouldNot be(null)
  }

  test("OpenVinoInferenceModel should predict correctly") {
    val indata1 = Source.fromFile(fasterrcnnInputdata1FilePath).getLines().map(_.toFloat).toArray
    val indata2 = Source.fromFile(fasterrcnnInputdata2FilePath).getLines().map(_.toFloat).toArray
    val outdata1 = Source.fromFile(fasterrcnnOutputdata1FilePath).getLines().map(_.toFloat).toArray
    val outdata2 = Source.fromFile(fasterrcnnOutputdata2FilePath).getLines().map(_.toFloat).toArray
    println(indata1.length, indata2.length, 1 * 3 * 600 * 600)
    println(outdata1.length, outdata2.length)
    val input1 = new JTensor(indata1, fasterrcnnInputShape)
    val input2 = new JTensor(indata2, fasterrcnnInputShape)
    val inputs = Arrays.asList(
      Arrays.asList({
        input1
      }),
      Arrays.asList({
        input2
      }))
    val results1 = fasterrcnnModel1.predict(inputs)
    assert(almostEqual(results1.get(0).get(0).getData, outdata1, 0.00001f))
    assert(almostEqual(results1.get(1).get(0).getData, outdata2, 0.00001f))

    val results2 = fasterrcnnModel2.predict(inputs)
    assert(almostEqual(results2.get(0).get(0).getData, outdata1, 0.00001f))
    assert(almostEqual(results2.get(1).get(0).getData, outdata2, 0.00001f))

    val results3 = fasterrcnnModel1.predict(inputs)
    assert(almostEqual(results3.get(0).get(0).getData, outdata1, 0.00001f))
    assert(almostEqual(results3.get(1).get(0).getData, outdata2, 0.00001f))

    val results4 = fasterrcnnModel2.predict(inputs)
    assert(almostEqual(results4.get(0).get(0).getData, outdata1, 0.00001f))
    assert(almostEqual(results4.get(1).get(0).getData, outdata2, 0.00001f))

    val threads = List.range(0, 5).map(i => {
      new Thread() {
        override def run(): Unit = {
          val results = fasterrcnnModel1.predict(inputs)
          assert(almostEqual(results.get(0).get(0).getData, outdata1, 0.00001f))
          assert(almostEqual(results.get(1).get(0).getData, outdata2, 0.00001f))
        }
      }
    })
    threads.foreach(_.start())
    threads.foreach(_.join())

    val threads2 = List.range(0, 5).map(i => {
      new Thread() {
        override def run(): Unit = {
          val results = fasterrcnnModel2.predict(inputs)
          assert(almostEqual(results.get(0).get(0).getData, outdata1, 0.00001f))
          assert(almostEqual(results.get(1).get(0).getData, outdata2, 0.00001f))
        }
      }
    })
    threads2.foreach(_.start())
    threads2.foreach(_.join())
  }

  def almostEqual(x: Float, y: Float, precision: Float): Boolean = (x - y).abs <= precision
  def almostEqual(x: Array[Float], y: Array[Float], precision: Float): Boolean = {
    x.length == y.length match {
      case true => x.zip(y).filter(t => !almostEqual(t._1, t._2, precision)).length == 0
      case false => false
    }
  }
}
