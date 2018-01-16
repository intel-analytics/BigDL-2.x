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

package com.intel.analytics.zoo.models.imageclassification.example

import java.util
import java.util.Random

import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.transform.vision.image.{ImageFrame, LocalImageFrame}
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.bigdl.zoo.models.Predictor
import com.intel.analytics.zoo.models.Configure
import com.intel.analytics.zoo.models.imageclassification.example.Predict.logger
import com.intel.analytics.zoo.models.imageclassification.util.LabelOutput
import org.apache.log4j.{Level, Logger}
import org.apache.storm.{Config, LocalCluster}
import org.apache.storm.spout.SpoutOutputCollector
import org.apache.storm.task.TopologyContext
import org.apache.storm.topology.{BasicOutputCollector, OutputFieldsDeclarer, TopologyBuilder}
import org.apache.storm.topology.base.{BaseBasicBolt, BaseRichSpout}
import org.apache.storm.tuple.{Fields, Tuple, Values}
import scopt.OptionParser

object PredictStreaming {

  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.zoo").setLevel(Level.INFO)

  val logger = Logger.getLogger(getClass)

  case class TopNClassificationParam(imageFolder: String = "",
                                     model: String = "",
                                     topN: Int = 5)

  val parser = new OptionParser[TopNClassificationParam]("ImageClassification demo") {
    head("Image Classification with BigDL and Storm")
    opt[String]('f', "folder")
      .text("where you put the demo image data")
      .action((x, c) => c.copy(imageFolder = x))
      .required()
    opt[String]("model")
      .text("BigDL model")
      .action((x, c) => c.copy(model = x))
      .required()
    opt[Int]("topN")
      .text("top N number")
      .action((x, c) => c.copy(topN = x))
      .required()
  }

  def main(args: Array[String]): Unit = {
    parser.parse(args, TopNClassificationParam()).foreach { param =>
      val imgFolder = param.imageFolder
      val modelPath = param.model
      val topN = param.topN

      val conf = new Config

      val builder = new TopologyBuilder
      builder.setSpout("spout", new ImageSpout(imgFolder))

      builder.setBolt("predict", new ImagePredictor(modelPath)).
        shuffleGrouping("spout")

      builder.setBolt("labelOutput", new LabelResult(modelPath)).
        shuffleGrouping("predict")

      builder.setBolt("echo", new EchoTopN(topN)).
        shuffleGrouping("labelOutput")
      conf.setDebug(true)
      conf.setNumWorkers(4)
      val cluster = new LocalCluster
      cluster.submitTopology("PredictStreaming", conf, builder.createTopology)
    }
  }
}


// A Spout implementation to simulate continously feeding data

class ImageSpout(val path: String) extends BaseRichSpout{

  private var collector : SpoutOutputCollector = null

  var imageFrame : ImageFrame = ImageFrame.read(path)

  val images = imageFrame.asInstanceOf[LocalImageFrame].array

  private var rand: Random = null

  override def declareOutputFields(outputFieldsDeclarer: OutputFieldsDeclarer): Unit = {
    outputFieldsDeclarer.declare(new Fields("img"))
  }

  override def nextTuple(): Unit = {
    Thread.sleep(1000)
    val image = images(rand.nextInt(images.length))
    val imageFram = ImageFrame.array(Array(image))
    collector.emit(new Values(imageFram))
  }

  override def open(map: util.Map[_, _],
                    topologyContext: TopologyContext,
                    spoutOutputCollector: SpoutOutputCollector): Unit = {
    collector = spoutOutputCollector
    rand = new Random
  }
}

class ImagePredictor(val modelPath: String) extends BaseBasicBolt {

  private var model : AbstractModule[Activity, Activity, Float] = null
  private var predictor : Predictor[Float] = null
  override def execute(tuple: Tuple, basicOutputCollector: BasicOutputCollector): Unit = {
    val imageFrame = tuple.getValue(0).asInstanceOf[ImageFrame]
    basicOutputCollector.emit(new Values(predictor.predict(imageFrame)))
  }

  override def declareOutputFields(outputFieldsDeclarer: OutputFieldsDeclarer): Unit = {
    outputFieldsDeclarer.declare(new Fields("img"))
  }

  override def prepare(stormConf: util.Map[_, _], context: TopologyContext): Unit = {
    model = Module.loadModule[Float](modelPath)
    predictor = Predictor(model)
    System.setProperty("bigdl.localMode", "true")
    System.setProperty("bigdl.coreNumber", 1.toString)
    Engine.init
  }
}

class LabelResult(val modelPath: String) extends BaseBasicBolt {
  private var labelOutput : LabelOutput = null
  override def execute(tuple: Tuple, basicOutputCollector: BasicOutputCollector): Unit = {
    val imageFrame = tuple.getValue(0).asInstanceOf[ImageFrame]
    val result = labelOutput(imageFrame)
    basicOutputCollector.emit(new Values(result))
  }

  override def declareOutputFields(outputFieldsDeclarer: OutputFieldsDeclarer): Unit = {
    outputFieldsDeclarer.declare(new Fields("img"))
  }

  override def prepare(stormConf: util.Map[_, _], context: TopologyContext): Unit = {
    val model = Module.loadModule[Float](modelPath)
    val labelMap = Configure.parse[Float](model.getName()).labelMap
    labelOutput = LabelOutput(labelMap, "clses", "probs")
    System.setProperty("bigdl.localMode", "true")
    System.setProperty("bigdl.coreNumber", 1.toString)
    Engine.init
  }
}

class EchoTopN(val topN: Int) extends BaseBasicBolt {

  override def execute(tuple: Tuple, basicOutputCollector: BasicOutputCollector): Unit = {
    val images = tuple.getValue(0).asInstanceOf[ImageFrame].toLocal().array
    logger.info(s"Prediction result")
    images.foreach(imageFeature => {
      logger.info(s"image : ${imageFeature.uri}, top ${topN}")
      val clsses = imageFeature("clses").asInstanceOf[Array[String]]
      val probs = imageFeature("probs").asInstanceOf[Array[Float]]
      for (i <- 0 until topN) {
        logger.info(s"\t class : ${clsses(i)}, credit : ${probs(i)}")
      }
    })

  }

  override def declareOutputFields(outputFieldsDeclarer: OutputFieldsDeclarer): Unit = {
    outputFieldsDeclarer.declare(new Fields("img"))
  }
}
