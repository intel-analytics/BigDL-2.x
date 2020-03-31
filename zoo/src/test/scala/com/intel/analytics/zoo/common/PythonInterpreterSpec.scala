package com.intel.analytics.zoo.common

import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import com.intel.analytics.zoo.pipeline.api.net.NetUtils
import jep.{NDArray, SharedInterpreter}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}

class PythonInterpreterSpec extends ZooSpecHelper{
  protected def ifskipTest(): Unit = {
    // Skip unitest if environment is not ready, PYTHONHOME should be set in environment
    if (System.getenv("PYTHONHOME") == null) {
      cancel("Please export PYTHONHOME before this test.")
    } else {
      logger.info(s"use python home: ${System.getenv("PYTHONHOME")}")
      Logger.getLogger(PythonInterpreter.getClass()).setLevel(Level.DEBUG)
    }
  }

  "interp" should "work in all thread" in {
    ifskipTest()
    val code =
      s"""
         |import numpy as np
         |a = np.array([1, 2, 3])
         |""".stripMargin
    PythonInterpreter.exec(code)
    println(PythonInterpreter.getValue[NDArray[_]]("a").getData())
    (0 until 1).toParArray.foreach{i =>
      println(Thread.currentThread())
      PythonInterpreter.exec(code)
    }
    val sc = SparkContext.getOrCreate(new SparkConf().setAppName("app").setMaster("local[4]"))
    (0 to 10).foreach(i =>
      sc.parallelize(0 to 10, 1).mapPartitions(i => {
        println(Thread.currentThread())
        PythonInterpreter.exec("a = np.array([1, 2, 3])")
        i
      }).count()
    )
  }
}
