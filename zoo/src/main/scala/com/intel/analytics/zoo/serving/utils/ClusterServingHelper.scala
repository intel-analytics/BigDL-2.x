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


package com.intel.analytics.zoo.serving.utils

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.EngineRef
import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.inference.InferenceModel
import scopt.OptionParser
import org.apache.log4j.Logger
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession

import scala.reflect.ClassTag

case class LoaderParams(modelType: String = null,
                        modelFolder: String = null,
                        batchSize: Int = 4,
                        isInt8: Boolean = false,
                        topN: Int = 1,
                        redis: String = "localhost:6379",
                        outputPath: String = "",
                        task: String = "image-classification",
                        classNum: Int = 5000,
                        nodeNum: Int = 1)

case class Result(id: String, value: String)

class ClusterServingHelper {

  val parser = new OptionParser[LoaderParams]("Cluster Serving") {

    opt[String]('t', "modelType")
      .text("Model type, could be caffe, keras")
      .action((x, c) => c.copy(modelType = x))

    opt[String]('f', "modelFolder")
      .text("weight file path")
      .action((x, c) => c.copy(modelFolder = x))
      .required()

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
    opt[String]('m', "task")
      .text("task name, e.g. image-classification")
      .action((x, c) => c.copy(task = x))
    opt[Int]('c', "classNum")
      .text("number of predicting classes")
      .action((x, c) => c.copy(classNum = x))
    opt[Int]('n', "nodeNum")
      .text("node number")
      .action((x, c) => c.copy(nodeNum = x))

  }

  var sc: SparkContext = null
  var params: LoaderParams = null
  var redisHost: String = null
  var redisPort: String = null
  var batchSize: Int = 4
  var topN: Int = 1
  var nodeNum: Int = 1

  var modelType: String = null
  var weightPath: String = null
  var defPath: String = null
  var dirPath: String = null
  var dummyMap: Map[Int, String] = Map()

  def initArgs(args: Array[String]): LoaderParams = {
    params = parser.parse(args, LoaderParams()).get
    require(params.redis.split(":").length == 2, "Your redis host " +
      "and port are not valid, please check.")
    redisHost = params.redis.split(":").head.trim
    redisPort = params.redis.split(":").last.trim
    batchSize = params.batchSize
    topN = params.topN
    nodeNum = params.nodeNum

    parseModelType(params.modelFolder)

    for (i <- 0 to params.classNum) {
      dummyMap += (i -> ("Class No." + i.toString))
    }
    params
  }

  def initContext(): Unit = {
    val conf = NNContext.createSparkConf().setAppName("Cluster Serving")
      .set("spark.redis.host", redisHost)
      .set("spark.redis.port", redisPort)
    sc = NNContext.initNNContext(conf)

  }

  /**
   * Inference Model do not use this method for model loading
   * This method is kept for future, not used now
   * @param ev
   * @tparam T
   * @return
   */
  def loadModel[T: ClassTag]()
                            (implicit ev: TensorNumeric[T]): RDD[Module[Float]] = {
    // deprecated
    val rmodel = modelType match {
      case "caffe" => Net.loadCaffe[Float](defPath, weightPath)
      case "tensorflow" => Net.loadTF[Float](weightPath)
      case "torch" => Net.loadTorch[Float](weightPath)
      case "bigdl" => Net.loadBigDL[Float](weightPath)
      case "keras" => Net.load[Float](weightPath)
    }
    val model = rmodel.quantize().evaluate()


    val bcModel = ModelBroadcast[Float]().broadcast(sc, model)
    val cachedModel = sc.range(1, 100, EngineRef.getNodeNumber())
      .coalesce(EngineRef.getNodeNumber())
      .mapPartitions(v => Iterator.single(bcModel.value(false, true))).cache()
    cachedModel
  }

  def loadInferenceModel(): Broadcast[InferenceModel] = {
    val model = new InferenceModel(1)
    modelType match {
      case "caffe" => model.doLoadCaffe(defPath, weightPath)
      case "tensorflow" => model.doLoadTF(weightPath)
      case "torch" => throw new Error("torch not supported in inference model")
      case "bigdl" => model.doLoad(weightPath)
      case "keras" => throw new Error("keras not supported in inference model")
      case "openvino" => model.doLoadOpenVINO(defPath, weightPath)
      case _ => throw new Error("Invalid model type, please check your model directory")
    }
    sc.broadcast(model)

  }

  def getSparkSession(): SparkSession = {
    SparkSession
      .builder
      .master(sc.master)
      .config("spark.redis.host", redisHost)
      .config("spark.redis.port", redisPort)
      .getOrCreate()
  }

  /**
   * To check if there already exists detected defPath or weightPath
   * @param defPath Boolean, true means need to check if it is not null
   * @param weightPath Boolean, true means need to check if it is not null
   */
  def throwOneModelError(modelType: Boolean,
                         defPath: Boolean, weightPath: Boolean): Unit = {

    if ((modelType && this.modelType != null) ||
        (defPath && this.defPath != null) ||
        (weightPath && this.weightPath != null)) {
      throw new Error("Only one model is allowed to exist in " +
        "model folder, please check your model folder to keep just" +
        "one model in the directory")

    }
  }

  def parseModelType(location: String): Unit = {

    import java.io.File
    val f = new File(location)
    val fileList = f.listFiles

    if (params.modelType == null) {

      for (file <- fileList) {
        val fName = file.getName
        val fPath = new File(location, fName).toString
        if (fName.endsWith("caffemodel")) {
          throwOneModelError(true, false, true)
          weightPath = fPath
          modelType = "caffe"
        }
        else if (fName.endsWith("prototxt")) {
          throwOneModelError(false, true, false)
          defPath = fPath
        }
        // ckpt seems not supported
        else if (fName.endsWith("pb")) {
          throwOneModelError(true, false, true)
          weightPath = location
          modelType = "tensorflow"
        }
        else if (fName.endsWith("t7")) {
          throwOneModelError(true, false, true)
          weightPath = fPath
          modelType = "torch"
        }
        else if (fName.endsWith("model")) {
          throwOneModelError(true, false, true)
          weightPath = fPath
          modelType = "bigdl"
        }
        else if (fName.endsWith("keras")) {
          throwOneModelError(true, false, true)
          weightPath = fPath
          modelType = "keras"
        }
        else if (fName.endsWith("bin")) {
          throwOneModelError(true, false, true)
          weightPath = fPath
          modelType = "openvino"
        }
        else if (fName.endsWith("xml")) {
          throwOneModelError(false, true, false)
          defPath = fPath
        }

      }
      if (modelType == null) throw new Error("You did not specify modelType before running" +
        " and the model type could not be inferred from the path" +
        "Note that you should put only one model in your model directory" +
        "And if you do not specify the modelType, it will be inferred " +
        "according to your model file extension name")
    }
    else {
      modelType = params.modelType.toLowerCase
    }
    println("model type is ", modelType, defPath, weightPath)
  }
  def getDummyMap(): Map[Int, String] = {
    return dummyMap
  }

}
