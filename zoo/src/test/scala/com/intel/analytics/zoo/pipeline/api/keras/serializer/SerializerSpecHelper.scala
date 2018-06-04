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

package com.intel.analytics.zoo.pipeline.api.keras.serializer

import java.io.File
import java.lang.reflect.Modifier

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import com.intel.analytics.bigdl.utils.serializer.{ModuleLoader, ModulePersister}
import com.intel.analytics.zoo.pipeline.api.keras.layers.NoKeras2
import org.reflections.Reflections
import org.reflections.scanners.SubTypesScanner
import org.reflections.util.{ClasspathHelper, ConfigurationBuilder, FilterBuilder}
import org.scalatest.{BeforeAndAfterAll, FlatSpec, Matchers, Tag}

import scala.collection.JavaConverters._
import scala.collection.mutable


abstract class SerializerSpecHelper extends FlatSpec with Matchers with BeforeAndAfterAll{

  val postFix = "bigdl"

  protected def runSerializationTest(
     module: AbstractModule[_, _, Float],
     input: Activity, cls: Class[_] = null) : Unit = {
    runSerializationTestWithMultiClass(module, input,
      if (cls == null) Array(module.getClass) else Array(cls))
  }

  protected def runSerializationTestWithMultiClass(
     module: AbstractModule[_, _, Float],
     input: Activity, classes: Array[Class[_]]) : Unit = {
    val name = module.getName
    val serFile = File.createTempFile(name, postFix)
    val originForward = module.evaluate().forward(input)

    ModulePersister.saveToFile[Float](serFile.getAbsolutePath, null, module.evaluate(), true)
    RNG.setSeed(1000)
    val loadedModule = ModuleLoader.loadFromFile[Float](serFile.getAbsolutePath)

    val afterLoadForward = loadedModule.forward(input)

    if (serFile.exists) {
      serFile.delete
    }

    afterLoadForward should be (originForward)
  }

  private val excluded = Set[String](
    "com.intel.analytics.zoo.pipeline.api.autograd.LambdaTorch",
    "com.intel.analytics.zoo.pipeline.api.net.TFNet")

  private val unRegularNameMapping = Map[String, String]()

  private val suffix = "SerialTest"

  protected def getPackagesForTest(): Set[String] = {
    Set("com.intel.analytics.zoo")
  }

  final def runTests(testClasses: mutable.HashSet[String], tag: Tag = NoKeras2): Unit = {
    testClasses.foreach(cls => {
      "Serialization test of module " + cls should "be correct" taggedAs(tag) in {
        val clsWholeName = getTestClassName(cls)
        try {
          val ins = Class.forName(clsWholeName)
          val testClass = ins.getConstructors()(0).newInstance()
          require(testClass.isInstanceOf[ModuleSerializationTest], s"$clsWholeName should be a " +
            s"subclass of com.intel.analytics.zoo.pipeline.api.keras.layers.serializer." +
            s"ModuleSerializationTest")
          testClass.asInstanceOf[ModuleSerializationTest].test()
        } catch {
          case t: Throwable => throw t
        }
      }
    })
  }

  final def getExpectedTests(): mutable.HashSet[String] = {
    val testClasses = new mutable.HashSet[String]()
    val filterBuilder = new FilterBuilder()
    val reflectionsBuilder = new ConfigurationBuilder()
      .filterInputsBy(filterBuilder)
    reflectionsBuilder.addUrls(ClasspathHelper.forPackage("com.intel.analytics.bigdl.nn"))
    getPackagesForTest().foreach {p =>
      reflectionsBuilder.addUrls(ClasspathHelper.forPackage(p))
    }

    reflectionsBuilder.setScanners(new SubTypesScanner())
    val reflections = new Reflections(reflectionsBuilder)

    val subTypes = reflections.getSubTypesOf(classOf[AbstractModule[_, _, _]]).asScala
      .filter(sub => !Modifier.isAbstract(sub.getModifiers))
      .filter(sub => !excluded.contains(sub.getName))
      .filter{sub =>
        !getPackagesForTest().filter{p => sub.getCanonicalName().contains(p)}
          .isEmpty
      }
    subTypes.foreach(sub => testClasses.add(sub.getName))
    testClasses
  }

  private def getTestClassName(clsName: String): String = {
    if (unRegularNameMapping.contains(clsName)) {
      unRegularNameMapping(clsName)
    } else {
      clsName + suffix
    }
  }
}
