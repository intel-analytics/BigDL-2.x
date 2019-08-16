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


package com.intel.analytics.zoo.utils

import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.EngineRef
import com.intel.analytics.zoo.pipeline.api.Net
import scopt.OptionParser
import org.apache.log4j.Logger
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession


import scala.reflect.ClassTag

case class LoaderParams(modelFolder: String = null,
                        batchSize: Int = 4,
                        isInt8: Boolean = false,
                        topN: Int = 1,
                        redis: String = "localhost:6379",
                        outputPath: String = "")



class Loader {

  val parser = new OptionParser[LoaderParams]("Zoo Serving") {

    //    opt[String]('t', "modelType")
    //      .text("Model type, could be caffe, keras")
    //      .action((x, c) => c.copy(modelType = x))
    //      .required()
    opt[String]('f', "modelFolder")
      .text("weight file path")
      .action((x, p) => p.copy(modelFolder = x))
      .required()
    //    opt[String]('d', "defPath")
    //      .text("prototxt file path if caffe model")
    //      .action((x, c) => c.copy(defPath = x))
    opt[String]('r', "redis")
      .text("redis url")
      .action((x, c) => c.copy(redis = x))

    opt[Int]('b', "batchSize")
      .text("Inference batch size")
      .action((x, c) => c.copy(batchSize = x))
    opt[String]('o', "outputPath")
      .text("redis url")
      .action((x, c) => c.copy(outputPath = x))
    opt[Int]('n', "topN")
      .text("number of return in classification task")
      .action((x, c) => c.copy(topN = x))


  }
  val logger = Logger.getLogger(getClass)
  var sc: SparkContext = null
  var params: LoaderParams = null
  var redisHost: String = null
  var redisPort: String = null
  var batchSize: Int = 4
  var topN: Int = 1

  var weightPath: String = null
  var defPath: String = null
  var dirPath: String = null
  //  var modelType: String = null

  def init(args: Array[String]) = {
    params = parser.parse(args, LoaderParams()).get

    redisHost = params.redis.split(":").head.trim
    redisPort = params.redis.split(":").last.trim
    batchSize = params.batchSize
    topN = params.topN

    val conf = NNContext.createSparkConf().setAppName("Redis Streaming Test")
      .set("spark.redis.host", redisHost)
      .set("spark.redis.port", redisPort)

    sc = NNContext.initNNContext(conf)

  }

  def loadModel[T: ClassTag]()
                            (implicit ev: TensorNumeric[T]) = {

    //    val model = if (modelType == "caffe") {
    //      val loadedModel = Module.loadCaffeModel[Float](params.defPath, params.weightPath)
    //      ModelConvertor.convert[Float](
    //        ModelConvertor.caffe2zoo(loadedModel), Boolean.box(false)).evaluate()
    //    } else {
    //      val loadedModel = Module.loadModule[Float](params.weightPath).quantize()
    //      loadedModel.evaluate()
    //    }
    val model = parseModelLocation(params.modelFolder)

    val sc = NNContext.initNNContext()
    val bcModel = ModelBroadcast[Float]().broadcast(sc, model)
    val cachedModel = sc.range(1, 100, EngineRef.getNodeNumber())
      .coalesce(EngineRef.getNodeNumber())
      .mapPartitions(v => Iterator.single(bcModel.value(false, true))).cache()
    cachedModel
  }

  def loadSparkSession(args: Array[String]) = {
    SparkSession
      .builder
      .master(sc.master)
      .config("spark.redis.host", redisHost)
      .config("spark.redis.port", redisPort)
      .getOrCreate()
  }
  def parseModelLocation(location: String):
  AbstractModule[Activity, Activity, Float] = {

    import java.io.File
    val f = new File(location)
    val fileList = f.listFiles
    var modelType: String = null
    var model: AbstractModule[Activity, Activity, Float] = null
    for (file <- fileList) {
      val fName = file.getName
      val fPath = new File(location, fName).toString
      if (fName.endsWith("caffemodel")) {
        weightPath = fPath
        modelType = "caffe"
      }
      else if (fName.endsWith("prototxt")) {
        defPath = fPath
      }
      else if (fName.endsWith("pb")) {
        weightPath = location
        modelType = "tensorflow"
      }
      else if (fName.endsWith("t7")) {
        weightPath = fPath
        modelType = "torch"
      }
      else if (fName.endsWith("model")) {
        weightPath = fPath
        modelType = "bigdl"
      }
      else if (fName.endsWith("keras")) {
        weightPath = fPath
        modelType = "keras"
      }
    }
    if (modelType == null) throw new RuntimeException("can not detect model type")
    model = modelType match {
      case "caffe" => Net.loadCaffe[Float](defPath, weightPath)
      case "tensorflow" => Net.loadTF[Float](weightPath)
      case "torch" => Net.loadTorch[Float](weightPath)
      case "bigdl" => Net.loadBigDL[Float](weightPath)
      case "keras" => Net.load[Float](weightPath)
    }
    model.evaluate()
  }

}
