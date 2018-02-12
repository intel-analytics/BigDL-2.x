package com.intel.analytics.zoo.pipeline.utils

import scala.io.Source

object LabelReader {

  /**
   * load coco label map
   */
  def readCocoLabelMap(): Map[Int, String] = {
    readLabelMap("/coco_classname.txt")
  }

  protected def readLabelMap(labelFileName: String): Map[Int, String] = {
    val labelFile = getClass().getResource(labelFileName)
    Source.fromURL(labelFile).getLines().zipWithIndex.map(x => (x._2, x._1)).toMap
  }
}
