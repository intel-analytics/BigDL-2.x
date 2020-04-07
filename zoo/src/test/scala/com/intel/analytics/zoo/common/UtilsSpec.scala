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

package com.intel.analytics.zoo.common

import com.intel.analytics.bigdl.utils.RandomGenerator
import org.apache.hadoop.fs.Path
import org.scalatest.{FlatSpec, Matchers}
import java.io.FileOutputStream

import com.intel.analytics.bigdl.dataset.Utils
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.python.api.{JTensor, PythonBigDL}
import com.intel.analytics.bigdl.utils.T
import org.apache.spark.bigdl.api.python.BigDLSerDe


class UtilsSpec extends FlatSpec with Matchers {
  val path: String = getClass.getClassLoader.getResource("qa").getPath
  val txtRelations: String = path + "/relations.txt"

  "Utils listFiles" should "work properly" in {
    val files = Utils.listPaths(path)
    assert(files.size == 3)
    val recursiveFiles = Utils.listPaths(path, true)
    assert(recursiveFiles.size == 13)
  }

  "Utils readBytes" should "work properly" in {
    val inputStream = Utils.open(txtRelations)
    val fileLen = inputStream.available()
    inputStream.close()
    val bytes = Utils.readBytes(txtRelations)
    assert(bytes.length == fileLen)
  }

  "Utils saveBytes" should "work properly" in {
    val fs = Utils.getFileSystem(path)
    // Generate random file
    val tmpFile = System.currentTimeMillis()
    val randomContent = new Array[Byte](1000)
    Utils.saveBytes(randomContent, path + "/" + tmpFile)
    // Delete random file
    fs.deleteOnExit(new Path(path + "/" + tmpFile))
    fs.close()
  }

  "Tensor pickle" should "work properly" in {
    val pickleFilename = "/home/yina/Documents" +
      "/testpickle.dat"
    val fos = new FileOutputStream(pickleFilename)
    val tensor = Tensor(T(
      T(6.1f, 5.2f, 4.3f),
      T(3.4f, 2.5f, 1.6f)))
    val jTensor = PythonBigDL.ofFloat().toJTensor(tensor)
    val tensorDump = BigDLSerDe.dumps(jTensor)
    fos.write(tensorDump)
    fos.flush()
    fos.close()
    System.out.println(tensorDump)
    val tensorLoad = BigDLSerDe.loads(tensorDump)
    if (tensorLoad.isInstanceOf[JTensor]) {
      val t = PythonBigDL.ofFloat().toTensor(tensorLoad.asInstanceOf[JTensor])
      print("aaa")
    }
    print("aaa")
  }
}
