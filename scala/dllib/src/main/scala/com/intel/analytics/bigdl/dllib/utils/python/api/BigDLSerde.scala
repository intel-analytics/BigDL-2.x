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

// Let the package prefix with "org.apache.spark" to access classes of Spark
// Some of the code originally from PySpark
package org.apache.spark.bigdl.api.python

import java.io.OutputStream
import java.nio.charset.StandardCharsets
import java.nio.{ByteBuffer, ByteOrder}
import java.util.{ArrayList => JArrayList, HashMap => JHashMap, List => JList, Map => JMap}

import com.intel.analytics.bigdl.python.api.{EvaluatedResult, JTensor, Sample}
import net.razorvine.pickle._
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.api.python.SerDeUtil
import org.apache.spark.mllib.api.python.SerDe
import org.apache.spark.rdd.RDD

import scala.collection.JavaConverters._
import scala.language.existentials
import scala.reflect.ClassTag

/**
 * Basic SerDe utility class.
 */
private[spark] abstract class BigDLSerDeBase {

  val PYSPARK_PACKAGE: String

  def initialize(): Unit

  def dumps(obj: AnyRef): Array[Byte] = {
    obj match {
      // Pickler in Python side cannot deserialize Scala Array normally. See SPARK-12834.
      case array: Array[_] => new Pickler().dumps(array.toSeq.asJava)
      case _ => new Pickler().dumps(obj)
    }
  }

  def loads(bytes: Array[Byte]): AnyRef = {
    new Unpickler().loads(bytes)
  }

  def asTupleRDD(rdd: RDD[Array[Any]]): RDD[(Int, Int)] = {
    rdd.map(x => (x(0).asInstanceOf[Int], x(1).asInstanceOf[Int]))
  }

  def fromTuple2RDD(rdd: RDD[(Any, Any)]): RDD[Array[Any]] = {
    rdd.map(x => Array(x._1, x._2))
  }


  def javaToPython(jRDD: JavaRDD[Any]): JavaRDD[Array[Byte]] = {
    jRDD.rdd.mapPartitions { iter =>
      initialize() // let it called in executor
      new SerDeUtil.AutoBatchedPickler(iter)
    }
  }


  def pythonToJava(pyRDD: JavaRDD[Array[Byte]], batched: Boolean)
  : JavaRDD[Any] = {
    pyRDD.rdd.mapPartitions { iter =>
      initialize()
      val unpickle = new Unpickler
      iter.flatMap { row =>
        val obj = unpickle.loads(row)
        if (batched) {
          obj match {
            case list: JArrayList[_] => list.asScala
            case arr: Array[_] => arr
          }
        } else {
          Seq(obj)
        }
      }
    }.toJavaRDD()
  }
}

/**
 * SerDe utility functions for BigDL.
 */
object BigDLSerDe extends BigDLSerDeBase with Serializable {

  val PYSPARK_PACKAGE = "bigdl.util.common"
  val LATIN1 = "ISO-8859-1"

  /**
   * Base class used for pickle
   */
  private[python] abstract class BigDLBasePickler[T: ClassTag]
    extends IObjectPickler with IObjectConstructor {

    val PYSPARK_PACKAGE = "bigdl.util.common"
    val LATIN1 = "ISO-8859-1"

    private val cls = implicitly[ClassTag[T]].runtimeClass
    println("cls.getname: " + cls.getName)
    private val module = PYSPARK_PACKAGE
    private val name = cls.getSimpleName

    def register(): Unit = {
      Pickler.registerCustomPickler(this.getClass, this)
      Pickler.registerCustomPickler(cls, this)
      Unpickler.registerConstructor(module, name, this)
      println(s"BigDLBasePickler registering: $module  $name")
    }

    def pickle(obj: Object, out: OutputStream, pickler: Pickler): Unit = {
      if (obj == this) {
        out.write(Opcodes.GLOBAL)
        out.write((module + "\n" + name + "\n").getBytes)
      } else {
        pickler.save(this) // it will be memorized by Pickler
        saveState(obj, out, pickler)
        out.write(Opcodes.REDUCE)
      }
    }

    private[python] def saveObjects(out: OutputStream, pickler: Pickler, objects: Any*) = {
      if (objects.length == 0 || objects.length > 3) {
        out.write(Opcodes.MARK)
      }
      objects.foreach(pickler.save)
      val code = objects.length match {
        case 1 => Opcodes.TUPLE1
        case 2 => Opcodes.TUPLE2
        case 3 => Opcodes.TUPLE3
        case _ => Opcodes.TUPLE
      }
      out.write(code)
    }

    protected def saveBytes(out: OutputStream, pickler: Pickler, bytes: Array[Byte]): Unit = {
      out.write(Opcodes.BINSTRING)
      out.write(PickleUtils.integer_to_bytes(bytes.length))
      out.write(bytes)
    }

    protected def getBytes(obj: Object): Array[Byte] = {
      if (obj.getClass.isArray) {
        obj.asInstanceOf[Array[Byte]]
      } else {
        // This must be ISO 8859-1 / Latin 1, not UTF-8, to interoperate correctly
        obj.asInstanceOf[String].getBytes(StandardCharsets.ISO_8859_1)
      }
    }

    protected def objToInt32Array(obj: Object): Array[Int] = {
      val bytes = getBytes(obj)
      val bb = ByteBuffer.wrap(bytes, 0, bytes.length)
      bb.order(ByteOrder.nativeOrder())
      val db = bb.asIntBuffer()
      val ans = new Array[Int](bytes.length / 4)
      db.get(ans)
      ans
    }

    protected def objToFloatArray(obj: Object): Array[Float] = {
      val bytes = getBytes(obj)
      val bb = ByteBuffer.wrap(bytes, 0, bytes.length)
      bb.order(ByteOrder.nativeOrder())
      val db = bb.asFloatBuffer()
      val ans = new Array[Float](bytes.length / 4)
      db.get(ans)
      ans
    }

    protected def floatArrayToBytes(arr: Array[Float]): Array[Byte] = {
      val bytes = new Array[Byte](4 * arr.size)
      val bb = ByteBuffer.wrap(bytes)
      bb.order(ByteOrder.nativeOrder())
      val db = bb.asFloatBuffer()
      db.put(arr)
      bytes
    }

    protected def int32ArrayToBytes(arr: Array[Int]): Array[Byte] = {
      val bytes = new Array[Byte](4 * arr.size)
      val bb = ByteBuffer.wrap(bytes)
      bb.order(ByteOrder.nativeOrder())
      val db = bb.asIntBuffer()
      db.put(arr)
      bytes
    }

    private[python] def saveState(obj: Object, out: OutputStream, pickler: Pickler)
  }

  private[python] class SamplePickler extends BigDLBasePickler[Sample] {

    def saveState(obj: Object, out: OutputStream, pickler: Pickler): Unit = {
      val record = obj.asInstanceOf[Sample]
      saveObjects(out,
        pickler,
        record.features,
        record.label,
        record.bigdlType)
    }

    def construct(args: Array[Object]): Object = {
      if (args.length != 3) {
        throw new PickleException("should be 3, not : " + args.length)
      }
      new Sample(args(0).asInstanceOf[JList[JTensor]],
        args(1).asInstanceOf[JTensor],
        args(2).asInstanceOf[String])
    }
  }

  private[python] class TestResultPickler extends BigDLBasePickler[EvaluatedResult] {

    def saveState(obj: Object, out: OutputStream, pickler: Pickler): Unit = {
      val testResult = obj.asInstanceOf[EvaluatedResult]
      saveObjects(out,
        pickler,
        testResult.result,
        testResult.totalNum, testResult.method)
    }

    def construct(args: Array[Object]): Object = {
      if (args.length != 3) {
        throw new PickleException("should be 3, not : " + args.length)
      }
      new EvaluatedResult(args(0).asInstanceOf[Float],
        args(1).asInstanceOf[Int],
        args(2).asInstanceOf[String])
    }
  }

  private[python] class JTensorPickler extends BigDLBasePickler[JTensor] {

    def saveState(obj: Object, out: OutputStream, pickler: Pickler): Unit = {
      val jTensor = obj.asInstanceOf[JTensor]
      saveBytes(out, pickler, floatArrayToBytes(jTensor.storage))
      saveBytes(out, pickler, int32ArrayToBytes(jTensor.shape))
      pickler.save(jTensor.bigdlType)
      out.write(Opcodes.TUPLE3)
    }


    def construct(args: Array[Object]): Object = {
      if (args.length != 3) {
        throw new PickleException("should be 3, not : " + args.length)
      }
      val bigdl_type = args(2).asInstanceOf[String]
      val storage = objToFloatArray(args(0))
      val shape = objToInt32Array(args(1))
      val result = new JTensor(storage, shape, bigdl_type)
      result
    }
  }

  var initialized = false

  override def initialize(): Unit = {
    synchronized {
      if (!initialized) {
        SerDe.initialize()
        new SamplePickler().register()
        new TestResultPickler().register()
        new JTensorPickler().register()
        initialized = true
      }
    }
  }
  // will not called in Executor automatically
  initialize()
}
