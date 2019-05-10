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

package com.intel.analytics.zoo.examples.streaming.objectdetection

import com.intel.analytics.zoo.common.Utils
import org.apache.hadoop.fs.Path
import org.apache.log4j.{Level, Logger}
import scopt.OptionParser

/*
  * Periodically write a text file (contains 16 image paths) to streamingPath
  */
object ImagePathWriter {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)

    val logger = Logger.getLogger(getClass)

    parser.parse(args, PathWriterParam()).foreach { params =>
      val lists = Utils.listPaths(params.imageSourcePath, false)
      lists.grouped(16).zipWithIndex.foreach { case (batch, id) =>
        val batchPath = new Path(params.streamingPath, id + ".txt").toString
        val dataOutStream = Utils.create(batchPath, true)
        try {
          batch.foreach(line => dataOutStream.writeBytes(line + "\n"))
        } finally {
          dataOutStream.close()
        }
        logger.info("wrote " + batchPath)
        Thread.sleep(4000)
      }
    }
  }

  private case class PathWriterParam(imageSourcePath: String = "",
                                     streamingPath: String = "file:///tmp/zoo/streaming")

  private val parser = new OptionParser[PathWriterParam]("PathWriterParam") {
    head("PathWriterParam")
    opt[String]("imageSourcePath")
      .text("folder that contains the source images, local file system only")
      .action((x, c) => c.copy(imageSourcePath = x))
      .required()
    opt[String]("streamingPath")
      .text("folder that used to store the streaming paths, i.e. file:///path")
      .action((x, c) => c.copy(streamingPath = x))
  }
}
