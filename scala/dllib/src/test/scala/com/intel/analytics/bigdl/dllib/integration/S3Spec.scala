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
package com.intel.analytics.bigdl.integration

import com.intel.analytics.bigdl.models.lenet.LeNet5
import com.intel.analytics.bigdl.models.resnet.Convolution
import com.intel.analytics.bigdl.nn.{Graph, Linear, Module, Sequential}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.caffe.{CaffeLoader, CaffePersister}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericDouble
import org.apache.commons.compress.utils.IOUtils
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.util.Random

@com.intel.analytics.bigdl.tags.Integration
class S3Spec extends FlatSpec with Matchers with BeforeAndAfter{
  val s3aPath = System.getProperty("s3aPath")

  private def processPath(path: String): String = {
    if (path.contains(":")) {
      path.substring(1)
    } else {
      path
    }
  }

  "save and load model from s3" should "be correct" in {
    val path = s3aPath + "/lenet.obj"
    val model = LeNet5(10)
    model.save(path, true)
    val lenet = Module.load(path)
    lenet should be (model)
  }

  "load caffe model from s3" should "work properly" in {
    val prototxt = getClass().getClassLoader().getResource("caffe/test.prototxt").getPath
    val modelPath = getClass().getClassLoader().getResource("caffe/test.caffemodel").getPath

    val s3Dir = s3aPath + s"/${ com.google.common.io.Files.createTempDir().getPath() }"

    def writeToS3(localFile: String, hdfsDir: String): Unit = {
      val src = new Path(localFile)
      val fs = src.getFileSystem(new Configuration(false))
      val inStream = fs.open(src)
      val dest = new Path(hdfsDir)
      val fsDest = dest.getFileSystem(new Configuration())
      val outFileStream = fsDest.create(dest)

      IOUtils.copy(inStream, outFileStream)

      // Close both files
      inStream.close()
      outFileStream.close()
    }

    writeToS3(prototxt, s3Dir + "/test.prototxt")
    writeToS3(modelPath, s3Dir + "/test.caffemodel")
    val module = Sequential()
      .add(Convolution(3, 4, 2, 2).setName("conv"))
      .add(Convolution(4, 3, 2, 2).setName("conv2"))
      .add(Linear(2, 27, withBias = false).setName("ip"))


    val model = CaffeLoader.load[Double](module, prototxt, modelPath)

    val modelFromS3 = CaffeLoader.load[Double](module, s3Dir + "/test.prototxt",
      s3Dir + "/test.caffemodel")

    model.getParameters() should be (modelFromS3.getParameters())

  }

  "Persist and Load Caffe to/from s3" should "works properly" in {

    val input1 = Tensor(10).apply1( e => Random.nextDouble())

    val input2 = Tensor()

    input2.resizeAs(input1).copy(input1)

    val linear = Linear(10, 10)

    // caffe only supports float, In order to compare the results, here we manually
    // set weight and bias to ensure there is no accurancy loss
    val weightTensor = Tensor(10, 10).fill(0.5)
    val biasTensor = Tensor(10).fill(0.1)
    linear.setWeightsBias(Array(weightTensor, biasTensor))

    val inputNode = linear.inputs()

    val graph = Graph(inputNode, inputNode)

    val hdfsDir = s3aPath + s"/${ com.google.common.io.Files.createTempDir().getPath() }"


    val res1 = graph.forward(input1)

    CaffePersister.persist(hdfsDir + "/test.prototxt", hdfsDir + "/test.caffemodel",
      graph, overwrite = true)

    val modelFromS3 = CaffeLoader.loadCaffe[Double](hdfsDir + "/test.prototxt",
      hdfsDir + "/test.caffemodel")._1

    val res2 = modelFromS3.forward(input2)

    res1 should be (res2)

  }
}
