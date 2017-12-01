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

package com.intel.analytics.zoo.pipeline.ssd.model

import java.io.File

import com.intel.analytics.bigdl.nn.Utils
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.zoo.pipeline.common.ModuleUtil
import com.intel.analytics.zoo.pipeline.common.caffe.{CaffeLoader, SSDCaffeLoader}
import org.scalatest.{FlatSpec, Matchers}

class SSDSpec extends FlatSpec with Matchers {
  val home = System.getProperty("user.home")

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

}
