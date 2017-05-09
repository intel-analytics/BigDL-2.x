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

package com.intel.analytics.zoo.pipeline.common.dataset

import java.io.File
import java.nio.file.Paths

import com.intel.analytics.zoo.pipeline.common.dataset.roiimage.{RoiByteImageToSeq, RoiImagePath}
import scopt.OptionParser

object RoiImageSeqGenerator {

  case class RoiImageSeqGeneratorParams(
    folder: String = ".",
    output: String = ".",
    parallel: Int = 1,
    blockSize: Int = 12800,
    imageSet: Option[String] = None
  )

  private val parser = new OptionParser[RoiImageSeqGeneratorParams]("Spark-DL Pascal VOC " +
    "Sequence File Generator") {
    head("Spark-DL Pascal VOC Sequence File Generator")
    opt[String]('f', "folder")
      .text("where you put the image data, if this is labeled pascal voc data, put the devkit path")
      .action((x, c) => c.copy(folder = x))
    opt[String]('o', "output folder")
      .text("where you put the generated seq files")
      .action((x, c) => c.copy(output = x))
    opt[Int]('p', "parallel")
      .text("parallel num")
      .action((x, c) => c.copy(parallel = x))
    opt[Int]('b', "blockSize")
      .text("block size")
      .action((x, c) => c.copy(blockSize = x))
    opt[String]('i', "imageSet")
      .text("image set, if this is the pascal voc or coco data," +
        " put the image set name, e.g. voc_2007_test or coco_testdev")
      .action((x, c) => c.copy(imageSet = Some(x)))
  }

  def main(args: Array[String]): Unit = {
    parser.parse(args, RoiImageSeqGeneratorParams()).map(param => {
      val roidbs = if (param.imageSet.isDefined) {
        Imdb.getImdb(param.imageSet.get, param.folder).getRoidb()
      } else {
        new File(param.folder).listFiles().map(f => RoiImagePath(f.getAbsolutePath))
      }

      val total = roidbs.length
      val iter = roidbs.toIterator

      (0 until param.parallel).map(tid => {
        val workingThread = new Thread(new Runnable {
          override def run(): Unit = {
            val fileIter = RoiByteImageToSeq(param.blockSize, Paths.get(param.output,
              s"$total-seq-$tid"))(iter)
            while (fileIter.hasNext) {
              println(s"Generated file ${ fileIter.next() }")
            }
          }
        })
        workingThread.setDaemon(false)
        workingThread.start()
        workingThread
      }).foreach(_.join())
    })
  }
}
