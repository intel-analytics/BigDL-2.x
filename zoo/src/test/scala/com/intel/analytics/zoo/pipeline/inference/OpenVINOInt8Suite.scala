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
import java.util.{Arrays, Properties}

import com.google.common.io.Files
import org.scalatest._
import org.slf4j.LoggerFactory
import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat

import scala.io.Source
import scala.language.postfixOps
import scala.sys.process._

@OpenVinoTest
class OpenVINOInt8Suite extends FunSuite with Matchers with BeforeAndAfterAll
  with InferenceSupportive {

  val s3Url = "https://s3-ap-southeast-1.amazonaws.com"

  var modelZooUrl = "http://download.tensorflow.org"
  try {
    val prop = new Properties()
    prop.load(this.getClass.getResourceAsStream("/app.properties"))
    modelZooUrl = prop.getProperty("data-store-url")
  } catch {
    case e: Exception =>
      modelZooUrl = "http://download.tensorflow.org"
  }

  val logger = LoggerFactory.getLogger(getClass)
  var tmpDir: File = _

  val resnet_v1_50_url = s"$modelZooUrl" + s"/models/resnet_v1_50_2016_08_28.tar.gz"
  val resnet_v1_50_tar = resnet_v1_50_url.split("/").last
  val resnet_v1_50_dir = resnet_v1_50_tar.replaceAll(".tar.gz", "")
  val resnet_v1_50_modelType = "resnet_v1_50"
  var resnet_v1_50_checkpointPath: String = _
  val resnet_v1_50_inputShape = Array(4, 224, 224, 3)
  val resnet_v1_50_ifReverseInputChannels = true
  val resnet_v1_50_meanValues = Array(123.68f, 116.78f, 103.94f)
  val resnet_v1_50_scale = 1.0f

  val calibrateValTarUrl = s"$s3Url/analytics-zoo-models/openvino/val_bmp_32.tar"
  val calibrateValTar = calibrateValTarUrl.split("/").last
  var calibrateValFilePath: String = _
  var calibrateValDirPath: String = _

  val resnet_v1_50_shape = Array(4, 3, 224, 224)
  val image_input_65_url = s"$s3Url/analytics-zoo-models/openvino/ic_input_65"
  val image_input_970_url = s"$s3Url/analytics-zoo-models/openvino/ic_input_970"
  var image_input_65_filePath: String = _
  var image_input_970_filePath: String = _

  val opencvLibTarURL = s"$s3Url/analytics-zoo-models/openvino/opencv_4.0.0_ubuntu_lib.tar"
  val opencvLibTar = opencvLibTarURL.split("/").last
  var opencvLibPath: String = _

  override def beforeAll() {
    tmpDir = Files.createTempDir()
    val dir = new File(s"${tmpDir.getAbsolutePath}/OpenVinoInt8Spec").getCanonicalPath

    s"wget -P $dir $resnet_v1_50_url" !;
    s"tar xvf $dir/$resnet_v1_50_tar -C $dir" !;

    s"wget -P $dir $calibrateValTarUrl" !;
    s"tar xvf $dir/$calibrateValTar -C $dir" !;

    s"wget -P $dir $image_input_65_url" !;
    s"wget -P $dir $image_input_970_url" !;

    s"wget -P $dir $opencvLibTarURL" !;
    s"tar xvf $dir/$opencvLibTar -C $dir" !;

    s"ls -alh $dir" !;

    resnet_v1_50_checkpointPath = s"$dir/resnet_v1_50.ckpt"
    calibrateValFilePath = s"$dir/val_bmp_32/val.txt"
    calibrateValDirPath = s"$dir/val_bmp_32/"

    image_input_65_filePath = s"$dir/ic_input_65"
    image_input_970_filePath = s"$dir/ic_input_970"

    opencvLibPath = s"$dir/lib"
  }

  override def afterAll() {
    // FileUtils.deleteDirectory(tmpDir)
    s"rm -rf $tmpDir" !;
  }


  test("openvino model should be optimized and calibrated") {
    InferenceModel.doOptimizeTF(
      null,
      resnet_v1_50_modelType,
      resnet_v1_50_checkpointPath,
      resnet_v1_50_inputShape,
      resnet_v1_50_ifReverseInputChannels,
      resnet_v1_50_meanValues,
      resnet_v1_50_scale,
      tmpDir.getAbsolutePath
    )
    val model_IR_path = s"${tmpDir.getAbsolutePath}/${resnet_v1_50_modelType}_inference_graph.xml"
    InferenceModel.doCalibrateTF(
      model_IR_path,
      "C",
      calibrateValFilePath,
      32,
      opencvLibPath,
      tmpDir.getAbsolutePath
    )
    tmpDir.listFiles().foreach(file => println(file.getAbsoluteFile))
  }

  test("openvino image classification model should load successfully") {
    val model = new InferenceModel(3)
    model.doLoadTF(null,
      resnet_v1_50_modelType,
      resnet_v1_50_checkpointPath,
      resnet_v1_50_inputShape,
      resnet_v1_50_ifReverseInputChannels,
      resnet_v1_50_meanValues,
      resnet_v1_50_scale)
    println(s"resnet_v1_50_model from tf loaded as $model")
    model shouldNot be(null)

    val indata1 = Source.fromFile(image_input_65_filePath).getLines().map(_.toFloat).toArray
    val indata2 = Source.fromFile(image_input_970_filePath).getLines().map(_.toFloat).toArray
    val data = indata1 ++ indata2 ++ indata1 ++ indata2
    val input1 = new JTensor(data, resnet_v1_50_shape)
    val input2 = new JTensor(data, resnet_v1_50_shape)
    val inputs = Arrays.asList(
      Arrays.asList({
        input1
      }),
      Arrays.asList({
        input2
      }))
    val results: util.List[util.List[JTensor]] = model.doPredict(inputs)
    val classes = results.toArray().map(list => {
      val inner = list.asInstanceOf[util.List[JTensor]].get(0)
      val class1 = inner.getData.slice(0, 1000).zipWithIndex.maxBy(_._1)._2
      val class2 = inner.getData.slice(1000, 2000).zipWithIndex.maxBy(_._1)._2
      (class1, class2)
    })
    println(classes.mkString(","))
  }

  test("openvino image classification model should load as calibrated successfully") {
    val model = new InferenceModel(3)
    model.doLoadTFAsCalibratedOpenVINO(null,
      resnet_v1_50_modelType,
      resnet_v1_50_checkpointPath,
      resnet_v1_50_inputShape,
      resnet_v1_50_ifReverseInputChannels,
      resnet_v1_50_meanValues,
      resnet_v1_50_scale,
      "C",
      calibrateValFilePath,
      32,
      opencvLibPath)
    println(s"resnet_v1_50_model from tf loaded as $model")
    model shouldNot be(null)
    val indata1 = Source.fromFile(image_input_65_filePath).getLines().map(_.toFloat).toArray
    val indata2 = Source.fromFile(image_input_970_filePath).getLines().map(_.toFloat).toArray
    val data = indata1 ++ indata2 ++ indata1 ++ indata2
    val input1 = new JTensor(data, resnet_v1_50_shape)
    val input2 = new JTensor(data, resnet_v1_50_shape)
    val inputs = Arrays.asList(
      Arrays.asList({
        input1
      }),
      Arrays.asList({
        input2
      }))
    val results: util.List[util.List[JTensor]] = model.doPredict(inputs)
    val classes = results.toArray().map(list => {
      val inner = list.asInstanceOf[util.List[JTensor]].get(0)
      val class1 = inner.getData.slice(0, 1000).zipWithIndex.maxBy(_._1)._2
      val class2 = inner.getData.slice(1000, 2000).zipWithIndex.maxBy(_._1)._2
      (class1, class2)
    })
    println(classes.mkString(","))
  }

  test("openvino resnet50 In8 should be calibrated and predictInt8(float) successfully") {
    val model = new InferenceModel(3)
    model.doLoadTFAsCalibratedOpenVINO(null,
      resnet_v1_50_modelType,
      resnet_v1_50_checkpointPath,
      resnet_v1_50_inputShape,
      resnet_v1_50_ifReverseInputChannels,
      resnet_v1_50_meanValues,
      resnet_v1_50_scale,
      "C",
      calibrateValFilePath,
      32,
      opencvLibPath)
    println(s"resnet_v1_50_model from tf loaded as $model")
    model shouldNot be(null)
    val indata1 = fromHWC2CHW(Source.fromFile(image_input_65_filePath)
      .getLines().map(_.toFloat).toArray)
    val indata2 = fromHWC2CHW(Source.fromFile(image_input_970_filePath)
      .getLines().map(_.toFloat).toArray)
    val labels = Array(65f, 970f)
    val data = indata1 ++ indata2 ++ indata1 ++ indata2
    val input1 = new JTensor(data, resnet_v1_50_shape)
    val input2 = new JTensor(data, resnet_v1_50_shape)
    val inputs = Arrays.asList(
      Arrays.asList({
        input1
      }),
      Arrays.asList({
        input2
      }))
    val results: util.List[util.List[JTensor]] = model.doPredictInt8(inputs)
    val classes = results.toArray().map(list => {
      val inner = list.asInstanceOf[util.List[JTensor]].get(0)
      val class1 = inner.getData.slice(0, 1000).zipWithIndex.maxBy(_._1)._2
      val class2 = inner.getData.slice(1000, 2000).zipWithIndex.maxBy(_._1)._2
      class1.toFloat
    })
    almostEqual(classes, labels, 0.1f)
    println(classes.mkString(","))
  }

  test("openvino resnet50 should predictInt image successfully") {
    val model = new InferenceModel(3)
    model.doLoadTF(null,
      resnet_v1_50_modelType,
      resnet_v1_50_checkpointPath,
      resnet_v1_50_inputShape,
      resnet_v1_50_ifReverseInputChannels,
      resnet_v1_50_meanValues,
      resnet_v1_50_scale)
    println(s"resnet_v1_50_model from tf loaded as $model")
    model shouldNot be(null)

    var indata1 = new Array[Float](3 * 224 * 224)
    var indata2 = new Array[Float](3 * 224 * 224)
    OpenCVMat.toFloatPixels(OpenCVMat.read(calibrateValDirPath +
      "ILSVRC2012_val_00000001.bmp"), indata1)
    OpenCVMat.toFloatPixels(OpenCVMat.read(calibrateValDirPath +
      "ILSVRC2012_val_00000002.bmp"), indata2)
    indata1 = fromHWC2CHW(indata1)
    indata2 = fromHWC2CHW(indata2)
    val labels = Array(65f, 970f)
    val data = indata1 ++ indata2 ++ indata1 ++ indata2
    val input1 = new JTensor(data, resnet_v1_50_shape)
    val input2 = new JTensor(data, resnet_v1_50_shape)
    val inputs = Arrays.asList(
      Arrays.asList({
        input1
      }),
      Arrays.asList({
        input2
      }))
    val results: util.List[util.List[JTensor]] = model.doPredict(inputs)
    val classes = results.toArray().map(list => {
      val inner = list.asInstanceOf[util.List[JTensor]].get(0)
      val class1 = inner.getData.slice(0, 1000).zipWithIndex.maxBy(_._1)._2
      val class2 = inner.getData.slice(1000, 2000).zipWithIndex.maxBy(_._1)._2
      class1.toFloat
    })
    almostEqual(classes, labels, 0.1f)
    println(classes.mkString(","))
  }

  test("openvino resnet50 In8 should predictInt8(float) image successfully") {
    val model = new InferenceModel(3)
    model.doLoadTFAsCalibratedOpenVINO(null,
      resnet_v1_50_modelType,
      resnet_v1_50_checkpointPath,
      resnet_v1_50_inputShape,
      resnet_v1_50_ifReverseInputChannels,
      resnet_v1_50_meanValues,
      resnet_v1_50_scale,
      "C",
      calibrateValFilePath,
      32,
      opencvLibPath)
    println(s"resnet_v1_50_model from tf loaded as $model")
    model shouldNot be(null)

    val indata1 = new Array[Float](3 * 224 * 224)
    val indata2 = new Array[Float](3 * 224 * 224)
    OpenCVMat.toFloatPixels(OpenCVMat.read(calibrateValDirPath +
      "ILSVRC2012_val_00000001.bmp"), indata1)
    OpenCVMat.toFloatPixels(OpenCVMat.read(calibrateValDirPath +
      "ILSVRC2012_val_00000002.bmp"), indata2)
    val labels = Array(65f, 970f)
    val data = indata1 ++ indata2 ++ indata1 ++ indata2
    val input1 = new JTensor(data, resnet_v1_50_shape)
    val input2 = new JTensor(data, resnet_v1_50_shape)
    val inputs = Arrays.asList(
      Arrays.asList({
        input1
      }),
      Arrays.asList({
        input2
      }))
    val results: util.List[util.List[JTensor]] = model.doPredictInt8(inputs)
    val classes = results.toArray().map(list => {
      val inner = list.asInstanceOf[util.List[JTensor]].get(0)
      val class1 = inner.getData.slice(0, 1000).zipWithIndex.maxBy(_._1)._2
      val class2 = inner.getData.slice(1000, 2000).zipWithIndex.maxBy(_._1)._2
      class1.toFloat
    })
    almostEqual(classes, labels, 0.1f)
    println(classes.mkString(","))
  }



  test("openvino should handle wrong batchSize correctly") {
    val model = new InferenceModel(3)
    model.doLoadTFAsCalibratedOpenVINO(null,
      resnet_v1_50_modelType,
      resnet_v1_50_checkpointPath,
      resnet_v1_50_inputShape,
      resnet_v1_50_ifReverseInputChannels,
      resnet_v1_50_meanValues,
      resnet_v1_50_scale,
      "C",
      calibrateValFilePath,
      32,
      opencvLibPath)
    println(s"resnet_v1_50_model from tf loaded as $model")
    model shouldNot be(null)
    val indata1 = Source.fromFile(image_input_65_filePath).getLines().map(_.toFloat).toArray
    val indata2 = Source.fromFile(image_input_970_filePath).getLines().map(_.toFloat).toArray
    val labels = Array(65f, 65f)
    // batchSize = 4, but given 3 and 5
    val data1 = indata1 ++ indata2 ++ indata1
    val data2 = indata2 ++ indata1 ++ indata2 ++ indata1 ++ indata2
    val input1 = new JTensor(data1, Array(3, 3, 224, 224))
    val input2 = new JTensor(data2, Array(5, 3, 224, 224))
    val inputs = Arrays.asList(
      Arrays.asList({
        input1
      }),
      Arrays.asList({
        input2
      }))
    val results: util.List[util.List[JTensor]] = model.doPredict(inputs)
    val classes = results.toArray().map(list => {
      val inner = list.asInstanceOf[util.List[JTensor]].get(0)
      val class1 = inner.getData.slice(0, 1000).zipWithIndex.maxBy(_._1)._2
      class1.toFloat
    })
    almostEqual(classes, labels, 0.1f)
    println(classes.mkString(","))
  }

  def fromHWC2CHW(data: Array[Float]): Array[Float] = {
    val tempArray = new Array[Float](3 * 224 * 224)
    for (h <- 0 to 223) {
      for (w <- 0 to 223) {
        for (c <- 0 to 2) {
          tempArray(c * h + w) = data(w * h + c)
        }
      }
    }
    return tempArray
  }

  def almostEqual(x: Float, y: Float, precision: Float): Boolean = {
    (x - y).abs <= precision match {
      case true => true
      case false => println(x, y); false
    }
  }

  def almostEqual(x: Array[Float], y: Array[Float], precision: Float): Boolean = {
    x.length == y.length match {
      case true => x.zip(y).filter(t => !almostEqual(t._1, t._2, precision)).length == 0
      case false => println(x.length, y.length); false
    }
  }
}
