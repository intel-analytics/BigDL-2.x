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

package com.intel.analytics.zoo.models.util


import scala.collection.mutable
import scala.io.Source
import scala.util.parsing.json.JSON

object ImageNetLableReader {

  val prototxt = getClass().getClassLoader().getResource("imageNetCls.lst").getPath

  val indexTolables = new mutable.HashMap[Int, String]()

  loadLables

  private def loadLables() : Unit = {
    val source: String = Source.fromFile(prototxt).getLines.mkString
    val content = JSON.parseFull(source).get.asInstanceOf[Map[String,String]]
    content.foreach{case (k, v) => {
      indexTolables(k.toInt) = v
    }}
  }

  def labelByIndex(index : Int) : String = {
    require(index >=1 && index <= 1000, "Index invalid")
    return indexTolables.get(index).get
  }

}
