///*
// * Copyright 2018 Analytics Zoo Authors.
// *
// * Licensed under the Apache License, Version 2.0 (the "License");
// * you may not use this file except in compliance with the License.
// * You may obtain a copy of the License at
// *
// *     http://www.apache.org/licenses/LICENSE-2.0
// *
// * Unless required by applicable law or agreed to in writing, software
// * distributed under the License is distributed on an "AS IS" BASIS,
// * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// * See the License for the specific language governing permissions and
// * limitations under the License.
// */
//
//package com.intel.analytics.zoo.examples.attention
//
//import com.intel.analytics.bigdl._
//import com.intel.analytics.bigdl.dataset.{DataSet, MiniBatch, Sample, SampleToMiniBatch}
//import com.intel.analytics.bigdl.nn.ClassNLLCriterion
//import com.intel.analytics.bigdl.numeric.NumericFloat
//import com.intel.analytics.bigdl.optim.{Optimizer, _}
//import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
//import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
//import com.intel.analytics.bigdl.utils.Shape
//import com.intel.analytics.zoo.common.NNContext
//import com.intel.analytics.zoo.models.attention.TransformerLayer
//import com.intel.analytics.zoo.pipeline.api.keras.layers.{Dense, Flatten, Input, Select}
//import com.intel.analytics.zoo.pipeline.api.keras.models.Model
//import com.intel.analytics.zoo.pipeline.api.keras.objectives.SparseCategoricalCrossEntropy
//import org.apache.log4j.{Level, Logger}
//import org.apache.spark.SparkConf
//import org.apache.spark.rdd.RDD
//import scopt.OptionParser
//
//import scala.io.Source
//import scala.reflect.ClassTag
//
//case class LocalParams(val inputDir: String = "/home/ding/data/chatbot/",
//                       val batchSize: Int = 1,
//                       val nEpochs: Int = 20
//                      )
//
//object transformer {
//
//  def main(args: Array[String]): Unit = {
//
//    val defaultParams = LocalParams()
//
//    val parser = new OptionParser[LocalParams]("bert Example") {
//      opt[String]("inputDir")
//        .text(s"inputDir")
//        .action((x, c) => c.copy(inputDir = x))
//      opt[Int]('b', "batchSize")
//        .text(s"batchSize")
//        .action((x, c) => c.copy(batchSize = x.toInt))
//      opt[Int]('e', "nEpochs")
//        .text("epoch numbers")
//        .action((x, c) => c.copy(nEpochs = x))
//    }
//
//    parser.parse(args, defaultParams).map {
//      params =>
//        run(params)
//    } getOrElse {
//      System.exit(1)
//    }
//  }
//
//  def run(param: LocalParams): Unit = {
//    Logger.getLogger("org").setLevel(Level.ERROR)
//    val conf = new SparkConf()
//    conf.setAppName("AttentionExample")
//    conf.set("spark.executor.extraJavaOptions", "-Xss512m")
//    conf.set("spark.driver.extraJavaOptions", "-Xss512m")
//    val sc = NNContext.initNNContext(conf)
//
//    val chat1 = Source
//      .fromFile(param.inputDir + "chat1_1.txt", "UTF-8")
//      .getLines
//      .toList
//      .map(_.split(",").map(_.toInt))
//
//    val tokens = sc.parallelize(chat1)
//
//    val trainRDD = tokens
//
//    val trainSet = trainRDD
//      .map(labeledChatToSample(_))
//
//    val transformer = TransformerLayer(8004)
//    val input = Input[Float](inputShape = Shape(3, 224, 224))
//    val feature = transformer.inputs(input)
//    val pool = Select(2, -1).inputs(feature)
//    val logits = Dense[Float](2).inputs(pool)
//
//    val model = Model[Float](input, logits)
//
//    val optimMethod = new SGD[Float]()
//    val endTrigger = Trigger.maxEpoch(1)
//
//    val optimizer = Optimizer(
//      model = model,
//      dataset = toDataSet(trainSet, batchSize = param.batchSize),
//      criterion = new ClassNLLCriterion[Float]()
//    )
//
//    optimizer
//      .setOptimMethod(optimMethod)
//      .setEndWhen(endTrigger)
//      .optimize()
//    sc.stop()
//  }
//
//  def toDataSet[T: ClassTag](x: RDD[Sample[T]], batchSize: Int)
//                            (implicit ev: TensorNumeric[T]): DataSet[MiniBatch[T]] = {
//    if (x != null) DataSet.rdd(x) -> SampleToMiniBatch[T](batchSize)
//    else null
//  }
//
//  def labeledChatToSample[T: ClassTag](
//                                        labeledChat: Array[Int])
//                                      (implicit ev: TensorNumeric[T]): Sample[T] = {
//
//    val data = labeledChat.map(ev.fromType(_))
//    val sentence1: Tensor[T] = Tensor(Storage(data))
//    val label: Tensor[T] = Tensor(Storage(Array(ev.fromType(1))))
//
//    val segmentIds = Tensor[T](sentence1.size())
//    val positionData = (0 until sentence1.size(1) toArray).map(ev.fromType(_))
//    val positionIds = Tensor[T](Storage(positionData))
//    val masks = Tensor[T](Array(1, 1, 1) ++ sentence1.size()).fill(ev.fromType(1))
//    Sample(featureTensors = Array(sentence1, segmentIds, positionIds, masks), labelTensor = label)
//  }
//}
