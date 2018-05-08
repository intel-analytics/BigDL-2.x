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

import java.util.Properties

import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.bigdl.utils.Engine.getClass
import org.apache.spark.{SparkConf, SparkContext, SparkException}
import org.apache.log4j.Logger
/**
 * [[NNContext]] wraps a spark context in Analytics Zoo.
 *
 */
object NNContext {

  private val logger = Logger.getLogger(getClass)

  /**
   * Gets a SparkContext with optimized configuration for BigDL performance. The method
   * will also initialize the BigDL engine.

   * Note: if you use spark-shell or Jupyter notebook, as the Spark context is created
   * before your code, you have to set Spark conf values through command line options
   * or properties file, and init BigDL engine manually.
   *
   * @param conf User defined Spark conf
   * @return Spark Context
   */
  def getNNContext(conf: SparkConf = null): SparkContext = {
    val bigdlConf = Engine.createSparkConf(conf)
    val sc = SparkContext.getOrCreate(bigdlConf)
    Engine.init
    checkSparkVersion(sc)
    checkScalaVersion()
    sc
  }

  private def checkVersion(
                            runtimeVersion: String,
                            compileTimeVersion: String,
                            project: String): Unit = {
    val Array(runtimeMajor, runtimeFeature, runtimeMaintance) =
      runtimeVersion.split("\\.").map(_.toInt)
    val Array(compileMajor, compileFeature, compileMaintance) =
      compileTimeVersion.split("\\.").map(_.toInt)
    if (!(runtimeMajor == compileMajor && runtimeFeature == compileFeature)) {
      throw new RuntimeException(s"The compile time $project version is not compatible with" +
        s" the Spark runtime version. Compile time version is $compileTimeVersion, runtime version" +
        s" is $runtimeVersion")
    }

    if (runtimeMaintance != compileMaintance) {
      logger.warn(s"The compile time $project version may not compatible with" +
        s" the Spark runtime version. Compile time version is $compileTimeVersion, runtime version" +
        s" is $runtimeVersion")
    }
  }


  private def checkSparkVersion(sc: SparkContext) = {
    checkVersion(sc.version, ZooBuildInfo.spark_version, "Spark")
  }

  private def checkScalaVersion() = {
    checkVersion(scala.util.Properties.versionString, ZooBuildInfo.scala_version, "Scala")
  }

}

private[zoo] object ZooBuildInfo {

  val (
    zoo_version: String,
    spark_version: String,
    scala_version: String,
    java_version: String) = {

    val resourceStream = Thread.currentThread().getContextClassLoader.
      getResourceAsStream("zoo-version-info.properties")

    try {
      val unknownProp = "<unknown>"
      val props = new Properties()
      props.load(resourceStream)
      (
        props.getProperty("zoo_version", unknownProp),
        props.getProperty("spark_version", unknownProp),
        props.getProperty("scala_version", unknownProp),
        props.getProperty("java_version", unknownProp)
      )
    } catch {
      case npe: NullPointerException =>
        throw new RuntimeException("Error while locating file zoo-version-info.properties, " +
          "if you are using an IDE to run your program, please make sure the maven generate-resources" +
          " phase is executed and a zoo-version-info.properties file is located in zoo/target/extra-resources", npe)
      case e: Exception =>
        throw new RuntimeException("Error loading properties from zoo-version-info.properties", e)
    } finally {
      if (resourceStream != null) {
        try {
          resourceStream.close()
        } catch {
          case e: Exception =>
            throw new SparkException("Error closing zoo build info resource stream", e)
        }
      }
    }
  }
}

