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

package com.intel.analytics.zoo.models.objectdetection.example

import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.transform.vision.image.ImageFrame
import com.intel.analytics.bigdl.utils.{Engine, File}
import com.intel.analytics.bigdl.zoo.models.Predictor
import com.intel.analytics.zoo.models.objectdetection.utils.Visualizer
import org.apache.log4j.{Level, Logger}
import Predict._
import scopt.OptionParser

object PredictLocal {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.zoo").setLevel(Level.INFO)

  val logger = Logger.getLogger(getClass)

  case class PredictLocal(image: String = "",
    outputFolder: String = "data/demo",
    model: String = "")

  val parser = new OptionParser[PredictLocal]("BigDL Object Detection Local Demo") {
    head("BigDL Object Detection Local Demo")
    opt[String]('i', "image")
      .text("where you put the demo image data, can be image folder or image path")
      .action((x, c) => c.copy(image = x))
      .required()
    opt[String]('o', "output")
      .text("where you put the output data")
      .action((x, c) => c.copy(outputFolder = x))
      .required()
    opt[String]("model")
      .text("BigDL model")
      .action((x, c) => c.copy(model = x))
  }

  def main(args: Array[String]): Unit = {
    parser.parse(args, PredictLocal()).foreach { params =>
      System.setProperty("bigdl.localMode", "true")
      Engine.init
      val model = Module.loadModule[Float](params.model)
      val data = ImageFrame.read(params.image)
      val predictor = Predictor(model)
      val output = predictor.predict(data)

      val visualizer = Visualizer(predictor.getLabelMap(), encoding = "jpg")
      val visualized = visualizer(output).toLocal()

      visualized.array.foreach(imageFeature => {
        File.saveBytes(imageFeature[Array[Byte]](Visualizer.visualized),
          getOutPath(params.outputFolder, imageFeature.uri(), "jpg"), true)
      })
      logger.info(s"labeled images are saved to ${params.outputFolder}")
    }
  }
}
