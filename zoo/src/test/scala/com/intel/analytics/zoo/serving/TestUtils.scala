package com.intel.analytics.zoo.serving

object TestUtils {
  /**
   *
   */
  def getStrFromResourceFile(path: String): String = {
    val resource = getClass().getClassLoader().getResource("serving")

    val dataPath = s"${resource.getPath}/$path"
    scala.io.Source.fromFile(dataPath).mkString
  }
}
