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

package com.intel.analytics.zoo.pipeline

import java.io.{File, FileNotFoundException}
import java.nio.file.Paths

import breeze.linalg.DenseMatrix
import breeze.numerics.abs
import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.Container
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.{Table, File => DlFile}

import scala.io.Source
import scala.reflect.ClassTag

object TestUtil {

  def printModuleTime(model: Module[Float], total: Double): Unit = {
    var moduleTime: Map[String, Long] = Map()

    def getModules(module: Module[Float]): Unit = {
      module match {
        case m: Container[_, _, Float] =>
          for (m <- module.asInstanceOf[Container[_, _, Float]].modules) getModules(m)
        case _ =>
          val name = module.getClass.getSimpleName
          if (moduleTime.contains(name)) {
            moduleTime += (name -> (moduleTime(name) + module.getTimes()(0)._2
              + module.getTimes()(0)._3))
          } else {
            moduleTime += (name -> (module.getTimes()(0)._2 + module.getTimes()(0)._3))
          }
      }
    }
    getModules(model)
    val sortedMod = moduleTime.toSeq.sortBy(- _._2)
    sortedMod.foreach(x => println(x._1 + "\t" + x._2 / 1e9 + s" ${(x._2 / 1e9 / total) * 100}%"))

  }

  def loadFeaturesFullName(s: String, hasSize: Boolean = true,
    middleRoot: String = middleRoot): Tensor[Float] = {
    loadFeaturesFullPath(Paths.get(middleRoot, s).toString, hasSize)
  }

  def loadFeaturesFullPath(s: String, hasSize: Boolean = true): Tensor[Float] = {
    println(s"load $s from file")

    if (hasSize) {
      val size = s.substring(s.lastIndexOf("-") + 1, s.lastIndexOf("."))
        .split("_").map(x => x.toInt)
      Tensor(Storage(Source.fromFile(s).getLines()
        .map(x => x.toFloat).toArray)).reshape(size)
    } else {
      Tensor(Storage(Source.fromFile(s).getLines()
        .map(x => x.toFloat).toArray))
    }
  }

  var middleRoot = "data/middle/vgg16/step1/"

  def loadFeatures(s: String, middleRoot: String = middleRoot)
  : Tensor[Float] = {
    if (s.contains(".txt")) {
      loadFeaturesFullName(s, hasSize = true, middleRoot)
    } else {
      val list = new File(middleRoot).listFiles()
      list.foreach(x => {
        if (x.getName.matches(s"$s-.*txt")) {
          return loadFeaturesFullName(x.getName, hasSize = true, middleRoot)
        }
      })
      throw new FileNotFoundException(s"cannot map $s")
    }
  }

  def assertEqual(expectedName: String, output: Tensor[Float], prec: Double,
    scale: Float = 1): Unit = {
    val expected = loadFeatures(expectedName).mul(scale)
    assertEqual2(expected, output, expectedName, prec)
  }

  def assertEqual2(expected: Tensor[Float], output: Tensor[Float],
    info: String = "", prec: Double): Unit = {
    if (!info.isEmpty) {
      println(s"compare $info ...")
    }
    if (expected.nElement() == 0 && output.nElement() == 0) {
      println(s"$info pass")
      return
    }
    require(expected.size().mkString(",") == output.size().mkString(","), "size mismatch: " +
      s"expected size ${expected.size().mkString(",")} " +
      s"does not match output ${output.size().mkString(",")}")
    expected.map(output, (a, b) => {
      if (Math.abs(a - b) > prec) {
        println(s"${Console.RED} ${a} does not equal ${b}, $info does not pass!!!!!!!!!!!!!!!!!!!" +
          s"${Console.WHITE} ")
        return
      }
      a
    })
    if (!info.isEmpty) {
      println(s"$info pass")
    }
  }

  def assertEqualTable[T: ClassTag](expected: Table, output: Table, info: String = ""): Unit = {
    require(expected.length() == output.length())
    (1 to expected.length()).foreach(i => assertEqualIgnoreSize(expected(i), output(i)))
  }

  def assertEqualIgnoreSize(expected: Tensor[Float], output: Tensor[Float], info: String = "",
    prec: Double = 1e-6): Unit = {
    if (!info.isEmpty) {
      println(s"compare $info ===============================")
    }
    require(expected.nElement() == output.nElement(), "size mismatch: " +
      s"expected size ${expected.size().mkString(",")} " +
      s"does not match output ${output.size().mkString(",")}")
    (expected.contiguous().storage().array() zip output.contiguous().storage().array()
      zip Stream.from(0)).foreach { x =>
      require(Math.abs(x._1._1 - x._1._2) < prec,
        s"expected ${x._1._1} does not equal actual ${x._1._2} in ${x._2}/${output.nElement()}")
    }
    if (!info.isEmpty) {
      println(s"$info pass")
    }
  }

  def existFile(f: String): Boolean = new java.io.File(f).exists()

  def load[M](filename: String): Option[M] = {
    try {
      if (existFile(filename)) return Some(DlFile.load[M](filename))
    } catch {
      case ex: Exception => None
    }
    None
  }

  def main(args: Array[String]): Unit = {
    val means = loadFeaturesFullPath("data/model/alexnet/means-3_256_256.txt")
    DlFile.save(means, "data/model/alexnet/means.obj")
  }

  def assertMatrixEqualTM(actual: Tensor[Float],
    expected: DenseMatrix[Double], diff: Double): Unit = {
    if (actual.dim() == 1) {
      assert(actual.nElement() == expected.size)
      var d = 1
      for (r <- 0 until expected.rows) {
        for (c <- 0 until expected.cols) {
          assert(abs(expected(r, c) - actual.valueAt(d)) < diff)
          d += 1
        }
      }
    } else {
      assert(actual.size(1) == expected.rows && actual.size(2) == expected.cols)
      for (r <- 0 until expected.rows) {
        for (c <- 0 until expected.cols) {
          assert(abs(expected(r, c) - actual.valueAt(r + 1, c + 1)) < diff)
        }
      }
    }
  }

  def assertMatrixEqualTM2(actual: Tensor[Float],
    expected: DenseMatrix[Float], diff: Double): Unit = {
    if (actual.dim() == 1) {
      assert(actual.nElement() == expected.size)
      var d = 1
      for (r <- 0 until expected.rows) {
        for (c <- 0 until expected.cols) {
          assert(abs(expected(r, c) - actual.valueAt(d)) < diff)
          d += 1
        }
      }
    } else {
      assert(actual.size(1) == expected.rows && actual.size(2) == expected.cols)
      for (r <- 0 until expected.rows) {
        for (c <- 0 until expected.cols) {
          assert(abs(expected(r, c) - actual.valueAt(r + 1, c + 1)) < diff,
            s"${expected(r, c)} not equal to ${actual.valueAt(r + 1, c + 1)}")
        }
      }
    }
  }


  def assertMatrixEqual(actual: DenseMatrix[Float],
    expected: DenseMatrix[Float], diff: Float): Unit = {
    for (r <- 0 until expected.rows) {
      for (c <- 0 until expected.cols) {
        assert(abs(expected(r, c) - actual(r, c)) < diff)
      }
    }
  }

  def assertMatrixEqualFD(actual: DenseMatrix[Float],
    expected: DenseMatrix[Double], diff: Double): Unit = {
    assert((actual.rows == expected.rows) && (actual.cols == expected.cols),
      s"actual shape is (${actual.rows}, ${actual.cols}), " +
        s"while expected shape is (${expected.rows}, ${expected.cols})")
    for (r <- 0 until expected.rows) {
      for (c <- 0 until expected.cols) {
        assert(abs(expected(r, c) - actual(r, c)) < diff)
      }
    }
  }


  def loadDataFromFile(fileName: String, sizes: Array[Int]): Tensor[Float] = {
    val lines = Source.fromFile(fileName).getLines().toArray.map(x => x.toFloat)
    Tensor(Storage(lines)).resize(sizes)
  }
}
