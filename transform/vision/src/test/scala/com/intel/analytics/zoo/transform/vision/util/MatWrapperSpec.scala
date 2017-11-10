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

package com.intel.analytics.zoo.transform.vision.util

import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.zoo.transform.vision.image.opencv.OpenCVMat
import com.intel.analytics.zoo.transform.vision.image.augmentation.{Crop, Resize}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.scalatest.{FlatSpec, Matchers}

class MatWrapperSpec extends FlatSpec with Matchers {
  "toBytes" should "work properly" in {
    val resource = getClass().getClassLoader().getResource("image/000025.jpg")
    val img = OpenCVMat.read(resource.getFile)
    Crop.transform(img, img, 0, 0, 0.5f, 0.5f)
    Resize.transform(img, img, 300, 300)

    val img2 = OpenCVMat.read(resource.getFile)
    Crop.transform(img2, img2, 0, 0, 0.5f, 0.5f)
    val bytes = OpenCVMat.toBytes(img2)
    val mat = OpenCVMat.toMat(bytes)
    Resize.transform(mat, mat, 300, 300)

    val floats1 = new Array[Float](3 * 300 * 300)
    val floats2 = new Array[Float](3 * 300 * 300)
    val buf = new OpenCVMat()
    OpenCVMat.toFloatBuf(img, floats1, buf)
    OpenCVMat.toFloatBuf(mat, floats2, buf)

    floats1.zip(floats2).foreach(x => {
      if (x._2 != x._1) {
        println(x._2 - x._1)
      }
    })
  }

  "serialize" should "work properly" in {
    val resource = getClass().getClassLoader().getResource("image/000025.jpg")
    val img = OpenCVMat.read(resource.getFile)
    println(img.width(), img.height())
    var sc: SparkContext = null

    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)
    val conf = Engine.createSparkConf().setMaster("local[2]")
      .setAppName("BigDL SSD Demo")
    sc = new SparkContext(conf)
    Engine.init
    val rdd = sc.parallelize(Array(img))
    rdd.foreach(mat => {
      println(mat.height(), mat.width())
    })
  }
}
