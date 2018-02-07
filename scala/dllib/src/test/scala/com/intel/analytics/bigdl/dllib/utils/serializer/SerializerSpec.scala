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
package com.intel.analytics.bigdl.utils.serializer

import java.lang.reflect.Modifier

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.utils.BigDLSpecHelper
import org.reflections.Reflections
import org.reflections.scanners.SubTypesScanner
import org.reflections.util.{ClasspathHelper, ConfigurationBuilder, FilterBuilder}

import collection.JavaConverters._
import scala.collection.mutable

class SerializerSpec extends BigDLSpecHelper {

  private val excluded = Set[String](
    "com.intel.analytics.bigdl.nn.CellUnit",
    "com.intel.analytics.bigdl.nn.tf.ControlDependency",
    "com.intel.analytics.bigdl.utils.tf.AdapterForTest",
    "com.intel.analytics.bigdl.utils.serializer.TestModule",
    "com.intel.analytics.bigdl.utils.ExceptionTest"
  )

  // Maybe one serial test class contains multiple module test
  // Also keras layer main/test class mapping are weired
  private val unRegularNameMapping = Map[String, String](
    // Many to one mapping
    "com.intel.analytics.bigdl.nn.ops.Enter" ->
      "com.intel.analytics.bigdl.nn.ops.ControlOpsSerialTest",
    "com.intel.analytics.bigdl.nn.ops.NextIteration" ->
      "com.intel.analytics.bigdl.nn.ops.ControlOpsSerialTest",
    "com.intel.analytics.bigdl.nn.ops.Exit" ->
      "com.intel.analytics.bigdl.nn.ops.ControlOpsSerialTest",
    "com.intel.analytics.bigdl.nn.ops.LoopCondition" ->
      "com.intel.analytics.bigdl.nn.ops.ControlOpsSerialTest",
    "com.intel.analytics.bigdl.nn.ops.StackCreator" ->
      "com.intel.analytics.bigdl.nn.ops.StackOpsSerialTest",
    "com.intel.analytics.bigdl.nn.ops.StackPush" ->
      "com.intel.analytics.bigdl.nn.ops.StackOpsSerialTest",
    "com.intel.analytics.bigdl.nn.ops.StackPop" ->
      "com.intel.analytics.bigdl.nn.ops.StackOpsSerialTest",

    // Keras layers
    "com.intel.analytics.bigdl.nn.keras.Dense" ->
      "com.intel.analytics.bigdl.keras.nn.DenseSerialTest"
  )

  private val suffix = "SerialTest"

  private val testClasses = new mutable.HashSet[String]()

  {
    val filterBuilder = new FilterBuilder()
    val reflections = new Reflections(new ConfigurationBuilder()
      .filterInputsBy(filterBuilder)
      .setUrls(ClasspathHelper.forPackage("com.intel.analytics.bigdl.nn"))
      .setScanners(new SubTypesScanner()))


    val subTypes = reflections.getSubTypesOf(classOf[AbstractModule[_, _, _]])
      .asScala.filter(sub => !Modifier.isAbstract(sub.getModifiers))
      .filter(sub => !excluded.contains(sub.getName))
    subTypes.foreach(sub => testClasses.add(sub.getName))
  }

  private def getTestClassName(clsName: String): String = {
    if (unRegularNameMapping.contains(clsName)) {
      unRegularNameMapping(clsName)
    } else {
      clsName + suffix
    }
  }

  testClasses.foreach(cls => {
    "Serialization test of module " + cls should "be correct" in {
      val clsWholeName = getTestClassName(cls)
      try {
        val ins = Class.forName(clsWholeName)
        val testClass = ins.getConstructors()(0).newInstance()
        require(testClass.isInstanceOf[ModuleSerializationTest], s"$clsWholeName should be a " +
          s"subclass of com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest")
        testClass.asInstanceOf[ModuleSerializationTest].test()
      } catch {
        case e: ClassNotFoundException =>
          cancel(s"Serialization test of module $cls has not " +
            s"been implemented. Please consider creating a serialization test class with name " +
            s"${clsWholeName} which extend com.intel.analytics.bigdl.utils.serializer." +
            s"ModuleSerializationTest")
        case t: Throwable => throw t
      }
    }
  })
}

private[bigdl] abstract class ModuleSerializationTest extends SerializerSpecHelper {
  def test(): Unit
}
