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

import java.io.{File, FileInputStream, FileWriter}
import java.nio.file.{Files, Path, Paths}

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.EngineRef
import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.inference.InferenceModel
import java.util.LinkedHashMap

import org.apache.log4j.Logger
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import org.yaml.snakeyaml.Yaml
import java.time.LocalDateTime

import scala.reflect.ClassTag

class ClusterServingHelper(_configPath: String = "config.yaml") {
  type HM = LinkedHashMap[String, String]

  val configPath = _configPath

  var lastModTime: String = null
  val logger: Logger = Logger.getLogger(getClass)
  val dateTime = LocalDateTime.now.toString

  var sc: SparkContext = null

  var modelInputs: String = null
  var modelOutputs: String = null
  var inferenceMode: String = null

  var redisHost: String = null
  var redisPort: String = null
  var nodeNum: Int = 1
  var coreNum: Int = 1
  var engineType: String = null
  var blasFlag: Boolean = false
  var chwFlag: Boolean = true

//  var dataType: Array[DataTypeEnumVal] = null
  var dataShape: Array[Array[Int]] = Array[Array[Int]]()
  var filter: String = null

  var logFile: FileWriter = null
  var logErrorFlag: Boolean = true
  var logSummaryFlag: Boolean = false

  /**
   * model related
   */
  var modelType: String = null
  var weightPath: String = null
  var defPath: String = null
  var modelDir: String = null
  /**
   * secure related
   */
  var redisSecureEnabled: Boolean = true
  var redisSecureTrustStorePath: String = null
  var redisSecureTrustStorePassword: String = null
  /**
   * Initialize the parameters by loading config file
   * create log file, set backend engine type flag
   * create "running" flag, for listening the stop signal
   */
  def initArgs(): Unit = {
    println("Loading config at ", configPath)
    val yamlParser = new Yaml()
    val input = new FileInputStream(new File(configPath))

    val configList = yamlParser.load(input).asInstanceOf[HM]

    // parse model field
    val modelConfig = configList.get("model").asInstanceOf[HM]
    modelDir = getYaml(modelConfig, "path", null).asInstanceOf[String]
    modelInputs = getYaml(modelConfig, "inputs", "").asInstanceOf[String]
    modelOutputs = getYaml(modelConfig, "outputs", "").asInstanceOf[String]
    inferenceMode = getYaml(modelConfig, "mode", "").asInstanceOf[String]

    parseModelType(modelDir)


    /**
     * reserved here to change engine type
     * engine type should be able to change in run time
     * but BigDL does not support this currently
     * Once BigDL supports it, engine type could be set here
     * And also other frameworks supporting multiple engine type
     */

    logFile = {
      val logF = new File("./cluster-serving.log")
      if (Files.exists(Paths.get("./cluster-serving.log"))) {
        logF.createNewFile()
      }
      new FileWriter(logF, true)
    }

    if (modelType.startsWith("tensorflow")) {
      chwFlag = false
    }
    // parse data field
    val dataConfig = configList.get("data").asInstanceOf[HM]
    val redis = getYaml(dataConfig, "src", "localhost:6379").asInstanceOf[String]
    require(redis.split(":").length == 2, "Your redis host " +
      "and port are not valid, please check.")
    redisHost = redis.split(":").head.trim
    redisPort = redis.split(":").last.trim

    val secureConfig = configList.get("secure").asInstanceOf[HM]
    redisSecureEnabled = if (getYaml(secureConfig, "secure_enabled", false) != null) {
      getYaml(secureConfig, "secure_enabled", false).asInstanceOf[Boolean]
    } else {
      false
    }

    val defaultPath = try {
      getClass.getClassLoader.getResource("keys/keystore.jks").getPath
    } catch {
      case _ => ""
    }
    redisSecureTrustStorePath = getYaml(
      secureConfig, "secure_trust_store_path", defaultPath)
      .asInstanceOf[String]
    redisSecureTrustStorePassword = getYaml(
      secureConfig, "secure_struct_store_password", "1234qwer").asInstanceOf[String]

    val shapeStr = getYaml(dataConfig, "shape", "3,224,224").asInstanceOf[String]
    require(shapeStr != null, "data shape in config must be specified.")
//    val shapeList = shape.split(",").map(x => x.trim.toInt)
//    for (i <- shapeList) {
//      dataShape = dataShape :+ i
//    }
    dataShape = ConfigUtils.parseShape(shapeStr.asInstanceOf[String])
    val typeStr = getYaml(dataConfig, "type", "image")
    require(typeStr != null, "data type in config must be specified.")
//    dataType = ConfigUtils.parseType(typeStr)


    filter = getYaml(dataConfig, "filter", "").asInstanceOf[String]

    val paramsConfig = configList.get("params").asInstanceOf[HM]
    coreNum = getYaml(paramsConfig, "core_number", 4).asInstanceOf[Int]

    if (modelType == "caffe" || modelType == "bigdl") {
      if (System.getProperty("bigdl.engineType", "mklblas")
        .toLowerCase() == "mklblas") {
        blasFlag = true
      }
      else blasFlag = false

    }
    else blasFlag = false

    new File("running").createNewFile()

  }

  /**
   * Check stop signal, return true if signal detected
   * @return
   */
  def checkStop(): Boolean = {
    if (!Files.exists(Paths.get("running"))) {
      return true
    }
    return false

  }

  /**
   * For dynamically update model, not used currently
   * @return
   */
  def updateConfig(): Boolean = {
    val lastModTime = Files.getLastModifiedTime(Paths.get(configPath)).toString
    if (this.lastModTime != lastModTime) {
      initArgs()
      this.lastModTime = lastModTime
      return true
    }
    return false
  }

  /**
   * The util of getting parameter from yaml
   * @param configList the hashmap of this field in yaml
   * @param key the key of target field
   * @param default default value used when the field is empty
   * @return
   */
  def getYaml(configList: HM, key: String, default: Any): Any = {
    val configValue: Any = try {
      configList.get(key)
    } catch {
      case _ => null
    }
    if (configValue == null) {
      if (default == null) throw new Error(configList.toString + key + " must be provided")
      else {
        return default
      }
    }
    else {
      println(configList.toString + key + " getted: " + configValue)
      if (configValue.isInstanceOf[Boolean] || configValue.isInstanceOf[Int]) {
        configValue
      } else {
        configValue.toString
      }
    }
  }

  /**
   * Initialize the Spark Context
   */
  def initContext(): Unit = {
    val conf = NNContext.createSparkConf().setAppName("Cluster Serving")
      .set("spark.redis.host", redisHost)
      .set("spark.redis.port", redisPort)
    sc = NNContext.initNNContext(conf)
    nodeNum = EngineRef.getNodeNumber()

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

  /**
   * Load inference model
   * The concurrent number of inference model depends on
   * backend engine type
   * @return
   */
  def loadInferenceModel(): InferenceModel = {
    val parallelNum = if (blasFlag) coreNum else 1
    val model = new InferenceModel(parallelNum)

    // Used for Tensorflow Model, it could not have intraThreadNum > 2^8
    // in some models, thus intraThreadNum should be limited
    val maxParallel = if (coreNum <= 64) {
      coreNum
    } else {
      64
    }

    modelType match {
      case "caffe" => model.doLoadCaffe(defPath, weightPath, blas = blasFlag)
      case "bigdl" => model.doLoadBigDL(weightPath, blas = blasFlag)
      case "tensorflowFrozenModel" =>
        model.doLoadTensorflow(weightPath, "frozenModel", maxParallel, 1, true)
      case "tensorflowSavedModel" =>
        modelInputs = modelInputs.filterNot((x: Char) => x.isWhitespace)
        modelOutputs = modelOutputs.filterNot((x: Char) => x.isWhitespace)
        val inputs = if (modelInputs == "") {
          null
        } else {
          modelInputs.split(",")
        }
        val outputs = if (modelOutputs == "") {
          null
        } else {
          modelOutputs.split(",")
        }
        model.doLoadTensorflow(weightPath, "savedModel", inputs, outputs)
      case "pytorch" => model.doLoadPyTorch(weightPath)
      case "keras" => logError("Keras currently not supported in Cluster Serving")
      case "openvino" => model.doLoadOpenVINO(defPath, weightPath, coreNum)
      case _ => logError("Invalid model type, please check your model directory")
    }
    model

  }

  /**
   * Get spark session for structured streaming
   * @return
   */
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
      logError("Only one model is allowed to exist in " +
        "model folder, please check your model folder to keep just" +
        "one model in the directory")

    }
  }

  /**
   * Log error message to local log file
   * @param msg
   */
  def logError(msg: String): Unit = {

    if (logErrorFlag) logFile.write(dateTime + " --- " + msg + "\n")
    throw new Error(msg)
  }


  /**
   * Infer the model type in model directory
   * Try every file in the directory, infer which are the
   * model definition file and model weight file
   * @param location
   */
  def parseModelType(location: String): Unit = {
    /**
     * Download file to local if the scheme is remote
     * Currently support hdfs, s3
     */
    val scheme = location.split(":").head
    val localModelPath = if (scheme == "file" || location.split(":").length <= 1) {
      location.split("file://").last
    } else {
      val path = Files.createTempDirectory("model")
      val dstPath = path.getParent + "/" + path.getFileName
      FileUtils.copyToLocal(location, dstPath)
      dstPath
    }

    /**
     * Initialize all relevant parameters at first
     */
    modelType = null
    weightPath = null
    defPath = null

    var variablesPathExist = false

    import java.io.File
    val f = new File(localModelPath)
    val fileList = f.listFiles

    // model type is always null, not support pass model type currently
    if (modelType == null) {

      for (file <- fileList) {
        val fName = file.getName
        val fPath = new File(localModelPath, fName).toString
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
          weightPath = localModelPath
          if (variablesPathExist) {
            modelType = "tensorflowSavedModel"
          } else {
            modelType = "tensorflowFrozenModel"
          }
        }
        else if (fName.endsWith("pt")) {
          throwOneModelError(true, false, true)
          weightPath = fPath
          modelType = "pytorch"
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
        else if (fName.equals("variables")) {
          if (modelType != null && modelType.equals("tensorflowFrozenModel")) {
            modelType = "tensorflowSavedModel"
          } else {
            variablesPathExist = true
          }
        }

      }
      if (modelType == null) logError("You did not specify modelType before running" +
        " and the model type could not be inferred from the path" +
        "Note that you should put only one model in your model directory" +
        "And if you do not specify the modelType, it will be inferred " +
        "according to your model file extension name")
    }
    else {
      modelType = modelType.toLowerCase
    }

  }

}
