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

package com.intel.analytics.zoo.examples.bert

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.models.bert.BERT
import com.intel.analytics.zoo.pipeline.api.keras.objectives.{SparseCategoricalCrossEntropy}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import scopt.OptionParser

import scala.io.Source
import scala.reflect.ClassTag

case class LocalParams(val inputDir: String = "/home/ding/data/chatbot/",
                       val batchSize: Int = 1,
                       val nEpochs: Int = 20
                      )

object bert {

  def main(args: Array[String]): Unit = {

    val defaultParams = LocalParams()

    val parser = new OptionParser[LocalParams]("bert Example") {
      opt[String]("inputDir")
        .text(s"inputDir")
        .action((x, c) => c.copy(inputDir = x))
      opt[Int]('b', "batchSize")
        .text(s"batchSize")
        .action((x, c) => c.copy(batchSize = x.toInt))
      opt[Int]('e', "nEpochs")
        .text("epoch numbers")
        .action((x, c) => c.copy(nEpochs = x))
    }

    parser.parse(args, defaultParams).map {
      params =>
        run(params)
    } getOrElse {
      System.exit(1)
    }
  }

  def run(param: LocalParams): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = new SparkConf()
    conf.setAppName("BertExample")
    conf.set("bigdl.ModelBroadcastFactory",
      "com.intel.analytics.bigdl.models.utils.ProtoBufferModelBroadcastFactory")
    conf.setExecutorEnv("bigdl.ModelBroadcastFactory",
      "com.intel.analytics.bigdl.models.utils.ProtoBufferModelBroadcastFactory")
    System.setProperty("bigdl.ModelBroadcastFactory",
      "com.intel.analytics.bigdl.models.utils.ProtoBufferModelBroadcastFactory")
    val sc = NNContext.initNNContext(conf)

    val chat1 = Source
      .fromFile(param.inputDir + "chat1_1.txt", "UTF-8")
      .getLines
      .toList
      .map(_.split(",").map(_.toInt))

    val tokens = sc.parallelize(chat1)

    val trainRDD = tokens

    val trainSet = trainRDD
      .map(labeledChatToSample(_))

    val model = BERT[Float](8004, debug = true, num_hidden_layers = 10)
    model.compile(optimizer = new RMSprop(learningRate = 0.001, decayRate = 0.9),
      loss = SparseCategoricalCrossEntropy[Float]())
    model.fit(trainSet, batchSize = param.batchSize, nbEpoch = param.nEpochs)
  }

  def labeledChatToSample[T: ClassTag](
    labeledChat: Array[Int])
    (implicit ev: TensorNumeric[T]): Sample[T] = {

    val data = labeledChat.map(ev.fromType(_))
    val sentence1: Tensor[T] = Tensor(Storage(data))
    val label: Tensor[T] = Tensor(Storage(Array(ev.fromType(1))))

    val segmentIds = Tensor[T](sentence1.size())
    val positionData = (0 until sentence1.size(1) toArray).map(ev.fromType(_))
    val positionIds = Tensor[T](Storage(positionData))
    val masks = Tensor[T](Array(1, 1, 1) ++ sentence1.size()).fill(ev.fromType(1))
    Sample(featureTensors = Array(sentence1, segmentIds, positionIds, masks), labelTensor = label)
  }
}
