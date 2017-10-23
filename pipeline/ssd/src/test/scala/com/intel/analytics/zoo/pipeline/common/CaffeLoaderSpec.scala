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

package com.intel.analytics.zoo.pipeline.common

import java.io.File

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.{File => DLFile}
import com.intel.analytics.zoo.pipeline.common.caffe.{CaffeLoader, SSDCaffeLoader, FrcnnCaffeLoader}
import com.intel.analytics.zoo.pipeline.ssd.TestUtil
import com.intel.analytics.zoo.pipeline.ssd.model.{SSDAlexNet, SSDVgg}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Engine, T}
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import com.intel.analytics.zoo.pipeline.common.nn.Proposal
import com.intel.analytics.zoo.pipeline.fasterrcnn.Postprocessor
import com.intel.analytics.zoo.pipeline.fasterrcnn.model.PostProcessParam
import org.apache.spark.SparkContext
import org.scalatest.{FlatSpec, Matchers}

class CaffeLoaderSpec extends FlatSpec with Matchers {
  val home = System.getProperty("user.home")
  "ssd caffe load dynamically" should "work properly" in {
    val prototxt = s"$home/Downloads/models/VGGNet/VOC0712/SSD_300x300/test.prototxt"
    val caffemodel = s"$home/Downloads/models/VGGNet/VOC0712/SSD_300x300/" +
      "VGG_VOC0712_SSD_300x300_iter_120000.caffemodel"
    if (!new File(prototxt).exists()) {
      cancel("local test")
    }

    val ssdcaffe = SSDCaffeLoader.loadCaffe(prototxt, caffemodel)

    val model = SSDVgg(21, 300)
    CaffeLoader.load(model, prototxt, caffemodel)
    RNG.setSeed(5000)
    val input = Tensor[Float](1, 3, 300, 300).rand(0, 255)

    ModuleUtil.shareMemory(ssdcaffe)
    ssdcaffe.evaluate()
    ssdcaffe.forward(input)

    model.evaluate()
    model.forward(input)

    val namedLayer1 = Utils.getNamedModules(ssdcaffe)

    val nameToLayer2 = Utils.getNamedModules(model)

    compare("relu9_1", nameToLayer2("relu9_1").output.toTensor[Float],
      namedLayer1("conv9_1_relu").output.toTensor[Float])

    compare("relu9_2", nameToLayer2("relu9_2").output.toTensor[Float],
      namedLayer1("conv9_2_relu").output.toTensor[Float])

    namedLayer1.keys.foreach(name => {
      if (nameToLayer2.contains(name)) {
        compare(name,
          namedLayer1(name).output.toTensor[Float], nameToLayer2(name).output.toTensor[Float])
      }
    })
    ssdcaffe.output should be(model.output)
  }

  def compare(name: String, t1: Tensor[Float], t2: Tensor[Float]): Boolean = {
    if (t1.toTensor[Float].nElement() != t2.toTensor[Float].nElement()) {
      println(s"compare ${name} fail, ${t1.toTensor[Float].nElement()} vs" +
        s" ${t2.toTensor[Float].nElement()}")
      return false
    }
    t1.toTensor[Float].map(t2.toTensor[Float],
      (a, b) => {
        if (Math.abs(a - b) > 1e-6) {
          println(s"compare $name fail")
          return false
        }
        a
      })
    true
  }

  "ssd caffe load dynamically 512" should "work properly" in {
    val prototxt = s"$home/Downloads/models/VGGNet/VOC0712/SSD_512x512/test.prototxt"
    val caffemodel = s"$home/Downloads/models/VGGNet/VOC0712/SSD_512x512/" +
      "VGG_VOC0712_SSD_512x512_iter_120000.caffemodel"

    if (!new File(prototxt).exists()) {
      cancel("local test")
    }
    val ssdcaffe = SSDCaffeLoader.loadCaffe(prototxt, caffemodel)
    val model = SSDVgg(21, 512)
    CaffeLoader.load(model, prototxt, caffemodel, true)

    ModuleUtil.shareMemory(ssdcaffe)

    RNG.setSeed(5000)
    val input = Tensor[Float](1, 3, 512, 512).rand(0, 255)

    ssdcaffe.evaluate()
    ssdcaffe.forward(input)

    model.evaluate()
    model.forward(input)

    val namedLayer1 = Utils.getNamedModules(ssdcaffe)

    val nameToLayer2 = Utils.getNamedModules(model)

    compare("relu9_1", nameToLayer2("relu9_1").output.toTensor[Float],
      namedLayer1("conv9_1_relu").output.toTensor[Float])

    compare("relu9_2", nameToLayer2("relu9_2").output.toTensor[Float],
      namedLayer1("conv9_2_relu").output.toTensor[Float])

    namedLayer1.keys.foreach(name => {
      if (nameToLayer2.contains(name)) {
        compare(name,
          namedLayer1(name).output.toTensor[Float], nameToLayer2(name).output.toTensor[Float])
      }
    })
    ssdcaffe.output should be(model.output)
  }

  "ssd caffe load dynamically alexnet" should "work properly" in {
    val prototxt = s"$home/data/ssd/jingdong/deploy.prototxt"
    val caffemodel = s"$home/data/ssd/jingdong/" +
      "ALEXNET_JDLOGO_V4_SSD_300x300_iter_920.caffemodel"

    if (!new File(prototxt).exists()) {
      cancel("local test")
    }
    val ssdcaffe = SSDCaffeLoader.loadCaffe(prototxt, caffemodel)

    ModuleUtil.shareMemory(ssdcaffe)
    val model = SSDAlexNet(2, 300)
    CaffeLoader.load(model, prototxt, caffemodel, true)

    RNG.setSeed(5000)
    val input = Tensor[Float](1, 3, 300, 300).rand(0, 255)

    ssdcaffe.evaluate()
    ssdcaffe.forward(input)

    model.evaluate()
    model.forward(input)

    val namedLayer1 = Utils.getNamedModules(ssdcaffe)

    val nameToLayer2 = Utils.getNamedModules(model)

    namedLayer1.keys.foreach(name => {
      if (nameToLayer2.contains(name)) {
        compare(name,
          namedLayer1(name).output.toTensor[Float], nameToLayer2(name).output.toTensor[Float])
      } else {
      }
    })
    ssdcaffe.output should be(model.output)
  }

  "deepbit model load" should "work properly" in {
    val prototxt = s"$home/Downloads/models/deploy16.prototxt"
    val caffemodel = s"$home/Downloads/models/DeepBit16_final_iter_1.caffemodel"

    if (!new File(prototxt).exists()) {
      cancel("local test")
    }
    val model = SSDCaffeLoader.loadCaffe(prototxt, caffemodel)

    ModuleUtil.shareMemory(model)
    model.evaluate()
    val input = Tensor[Float](1, 3, 224, 224)
    val output = model.forward(input)
    println(output.toTensor[Float].size().mkString("x"))
  }

  "deepbit model load with caffe input" should "work properly" in {
    val prototxt = s"$home/Downloads/models/deploy16.prototxt"
    val caffemodel = s"$home/Downloads/models/DeepBit16_final_iter_1.caffemodel"

    if (!new File(prototxt).exists()) {
      cancel("local test")
    }
    TestUtil.middleRoot = "$home/data/deepbit"
    val input = TestUtil.loadFeaturesFullPath("$home/data/deepbit/data-2_3_224_224.txt")
    val model = CaffeLoader.loadCaffe[Float](prototxt, caffemodel)._1

    ModuleUtil.shareMemory(model)
    model.evaluate()
    val output = model.forward(input).toTensor[Float]
    val namedModule = Utils.getNamedModules(model)
    TestUtil.assertEqual("fc8_kevin", namedModule("fc8_kevin").output.toTensor, 1e-5)
    println(output.toTensor[Float].size().mkString("x"))
  }

  "deepbit 1.0 model load with caffe input" should "work properly" in {
    val prototxt = s"$home/Downloads/models/deepbit_1.0.prototxt"
    val caffemodel = s"$home/Downloads/models/deepbit_1.0.caffemodel"

    if (!new File(prototxt).exists()) {
      cancel("local test")
    }
    TestUtil.middleRoot = "$home/data/deepbit/1.0"
    val input = TestUtil.loadFeatures("data")
    //    val input = Tensor[Float](1, 3, 227, 227)
    val model = CaffeLoader.loadCaffe[Float](prototxt, caffemodel)._1

    ModuleUtil.shareMemory(model)
    model.evaluate()
    val output = model.forward(input).toTensor[Float]
    val namedModule = Utils.getNamedModules(model)
    TestUtil.assertEqual("fc8_kevin", namedModule("fc8_kevin").output.toTensor, 1e-7)
    println(output.toTensor[Float].size().mkString("x"))
  }

  "fasterrcnn vgg load with caffe" should "work properly" in {
    val conf = Engine.createSparkConf().setMaster("local[2]")
      .setAppName("Spark-DL Faster RCNN Test")
    val sc = new SparkContext(conf)
    Engine.init
    val prototxt = s"$home/data/caffeModels/vgg16/test.prototxt"
    val caffemodel = s"$home/data/caffeModels/vgg16/test.caffemodel"

    if (!new File(prototxt).exists()) {
      cancel("local test")
    }
    val model = FrcnnCaffeLoader.loadCaffe(prototxt, caffemodel, Array("proposal", "im_info"))
      .asInstanceOf[Graph[Float]]
    val input = T()
    input.insert(Tensor[Float](1, 3, 60, 90))
    input.insert(Tensor[Float](T(60, 90, 1, 1)).resize(1, 4))

    val postprocessor = Postprocessor(PostProcessParam(0.3f, 21, false, 100, 0.05))
    ModuleUtil.shareMemory(model)
    val modelWithPostprocess = Sequential[Float]().add(model).add(postprocessor)
    modelWithPostprocess.saveModule("/tmp/vgg.frcnn", true)
    modelWithPostprocess.forward(input)
//    println(model.output.toTable.length())
//    (1 to model.output.toTable.length()).foreach(i => {
//      println(model.output.toTable[Tensor[Float]](i).size().mkString("x"))
//    })
    // model.saveGraphTopology("/tmp/summary")
  }

  "pvanet load" should "work properly" in {

    val conf = Engine.createSparkConf().setMaster("local[2]")
      .setAppName("Spark-DL Faster RCNN Test")
    val sc = new SparkContext(conf)
    Engine.init
    val prototxt = s"$home/data/caffeModels/pvanet/test.prototxt"
    val caffemodel = s"$home/data/caffeModels/pvanet/PVA9.1_ImgNet_COCO_VOC0712.caffemodel"

    if (!new File(prototxt).exists()) {
      cancel("local test")
    }
    val model = FrcnnCaffeLoader.loadCaffe(prototxt, caffemodel, Array("proposal", "im_info"))
      .asInstanceOf[Graph[Float]]
    //    val model = DLFile.load[Graph[Float]]("/tmp/pvanet.bin")
    val input = T()
    input.insert(Tensor[Float](1, 3, 640, 960))
    input.insert(Tensor[Float](T(640, 960, 1, 1)).resize(1, 4))
//    model.save("/tmp/pvanet.bin", true)
//    model.saveModule("/tmp/pvanet.model", true)
    println("save model done")

    //    model.saveGraphTopology("/tmp/summary")
    val postprocessor = Postprocessor(PostProcessParam(0.4f, 21, true, 100, 0.05))
    ModuleUtil.shareMemory(model)
    val modelWithPostprocess = Sequential[Float]().add(model).add(postprocessor)

    modelWithPostprocess.saveModule("/tmp/pvanet.model", true)
    modelWithPostprocess.forward(input)
//    model.saveGraphTopology("/tmp/summary")
  }

  "pvanet forward" should "work properly" in {
    val conf = Engine.createSparkConf().setMaster("local[2]")
      .setAppName("Spark-DL Faster RCNN Test")
    val sc = new SparkContext(conf)
    Engine.init
    val prototxt = s"$home/data/caffeModels/pvanet/faster_rcnn_train_test_21cls.pt"
    val caffemodel = s"$home/data/caffeModels/pvanet/PVA9.1_ImgNet_COCO_VOC0712.caffemodel"

    if (!new File(prototxt).exists()) {
      cancel("local test")
    }
    val input = T()
    input.insert(TestUtil.loadFeaturesFullPath(s"$home/data/middle/pvanew/data-1_3_640_960.txt"))
    input.insert(Tensor[Float](T(640, 960, 1.9199999, 1.9199999)).resize(1, 4))
    val modelWithPostprocess = Module.loadModule[Float]("/tmp/pvanet.model")
    modelWithPostprocess.evaluate()
    println("load done !")
    TestUtil.middleRoot = s"$home/data/middle/pvanew/"
    modelWithPostprocess.forward(input)
    def assertEqual(name: String): Unit = {
      TestUtil.assertEqual(name, modelWithPostprocess(name).get.output.toTensor[Float], 1e-4)
    }
    def assertEqual2(name: String, name2: String): Unit = {
      TestUtil.assertEqual(name, modelWithPostprocess(name2).get.output.toTensor[Float], 1e-4)
    }
    assertEqual2("conv1_1_conv", "conv1_1/bn")
    assertEqual("pool1")
    assertEqual2("conv2_1_proj", "conv2_1/proj")
    assertEqual2("conv2_1_3", "conv2_1/3/conv")
    assertEqual("conv2_1")
    assertEqual("conv3_4")
    assertEqual("conv4_4")
    assertEqual("concat")
    assertEqual("upsample")
    assertEqual("downsample")
    assertEqual("roi_pool_conv5")
    assertEqual2("rois", "proposal")
    TestUtil.assertEqual("cls_prob", modelWithPostprocess("cls_prob").get.output.toTensor[Float], 1e-5)
    TestUtil.assertEqual("bbox_pred", modelWithPostprocess("bbox_pred").get.output.toTensor[Float], 1e-5)
//    model.saveGraphTopology("/tmp/summary")
  }

  "load and save" should "work" in {
    val model = Sequential[Float]()
    val graph = Input[Float]()
    val linear = Linear[Float](20, 20)
    linear.inputs(graph)
    model.add(linear)
    model.saveModule("/tmp/model", true)
    Module.loadModule[Float]("/tmp/model")
  }

  "load and save2" should "work" in {
    val model = Proposal(1, 1, Array[Float](0.1f), Array[Float](1f))
    model.saveModule("/tmp/propsal", true)
    Module.loadModule[Float]("/tmp/propsal")
  }

  "ev.scale" should "work" in {
    val array = new Array[Float](1228800)
    TensorNumeric.NumericFloat.scal(1, 4096, array, 1883279, 1)
  }

  "cmul" should "not dump" in {
    val cmul = CMul[Float](Array[Int](1, 4096, 1, 1))
    cmul.weight.randn()
    val input = Tensor[Float](300, 4096).randn()
    cmul.forward(input)
  }
}
