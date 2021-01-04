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

import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.EngineRef
import com.intel.analytics.zoo.pipeline.inference.InferenceModel
import java.util.{LinkedHashMap, UUID}

import org.yaml.snakeyaml.Yaml
import java.util

import com.intel.analytics.zoo.serving.ClusterServing
import org.apache.flink.core.execution.JobClient
import redis.clients.jedis.Jedis

import scala.collection.JavaConverters._
import scala.reflect.ClassTag

/**
 * The helper of Cluster Serving
 * by default, all parameters are loaded by config including model directory
 * However, in some condition, models are distributed to remote machine
 * and locate in tmp directory, but other configs are still needed.
 * Thus model directory could be passed and overwrite that in config YAML
 * @param _configPath the path of Cluster Serving config YAML
 * @param _modelDir the path of model, if null, will read from config YAML
 */
class ClusterServingHelper(_configPath: String = "config.yaml", _modelDir: String = null)
  extends Serializable {
  type HM = LinkedHashMap[String, String]

  val configPath = _configPath
  var jobName: String = _

  var lastModTime: String = null


  var redisHost: String = null
  var redisPort: Int = _
  var redisTimeout: Int = 5000
  var nodeNum: Int = 1
  var coreNum: Int = 1
  var modelPar: Int = 1
  var blasFlag: Boolean = false
  var chwFlag: Boolean = true

  var filter: String = null
  var resize: Boolean = false

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
  var redisSecureEnabled: Boolean = false
  var redisSecureTrustStorePath: String = null
  var redisSecureTrustStoreToken: String = null

  var modelEncrypted: Boolean = false

  /**
   * Initialize the parameters by loading config file
   * create log file, set backend engine type flag
   * create "running" flag, for listening the stop signal
   */
  def loadConfig(): Unit = {
    val yamlParser = new Yaml()
    val input = new FileInputStream(new File(configPath))

    val configList = yamlParser.load(input).asInstanceOf[HM]

    // parse model field
    val modelConfig = configList.get("model").asInstanceOf[HM]
    modelDir = if (_modelDir == null) {
      getYaml(modelConfig, "path", null).asInstanceOf[String]
    } else {
      _modelDir
    }
    jobName = getYaml(modelConfig,
      "name", Conventions.SERVING_STREAM_DEFAULT_NAME).asInstanceOf[String]



    /**
     * Tensorflow usually use NHWC input
     * While others use NCHW
     */

    // parse data field
    val dataConfig = configList.get("data").asInstanceOf[HM]
    val redis = getYaml(dataConfig, "src", "localhost:6379").asInstanceOf[String]
    require(redis.split(":").length == 2, "Your redis host " +
      "and port are not valid, please check.")
    redisHost = redis.split(":").head.trim
    redisPort = redis.split(":").last.trim.toInt

    val secureConfig = configList.get("secure").asInstanceOf[HM]
    redisSecureEnabled = getYaml(secureConfig, "secure_enabled", false).asInstanceOf[Boolean]

    val defaultPath = try {
      getClass.getClassLoader.getResource("keys/keystore.jks").getPath
    } catch {
      case _ => ""
    }
    redisSecureTrustStorePath = getYaml(
      secureConfig, "secure_trust_store_path", defaultPath)
      .asInstanceOf[String]
    redisSecureTrustStoreToken = getYaml(
      secureConfig, "secure_struct_store_password", "1234qwer").asInstanceOf[String]
    modelEncrypted = getYaml(secureConfig, "model_encrypted", false).asInstanceOf[Boolean]

    val typeStr = getYaml(dataConfig, "type", "image")
    require(typeStr != null, "data type in config must be specified.")

    filter = getYaml(dataConfig, "filter", "").asInstanceOf[String]
    resize = getYaml(dataConfig, "resize", true).asInstanceOf[Boolean]

    val paramsConfig = configList.get("params").asInstanceOf[HM]
    coreNum = getYaml(paramsConfig, "core_number", 4).asInstanceOf[Int]

    val modelParDefault = if (modelType == "openvino") coreNum else coreNum
    modelPar = getYaml(paramsConfig, "model_number", default = modelParDefault).asInstanceOf[Int]


    if (modelType == "caffe" || modelType == "bigdl") {
      if (System.getProperty("bigdl.engineType", "mklblas")
        .toLowerCase() == "mklblas") {
        blasFlag = true
      }
      else blasFlag = false
    }
    else blasFlag = false

    val redisConfig = configList.get("redis").asInstanceOf[HM]
    redisTimeout = getYaml(redisConfig, "timeout", 5000).asInstanceOf[Int]
    parseModelType(modelDir)
  }

  /**
   * To check if one of the running jobs already have this name
   * If yes, existed name is not allow to used, will not submit
   * the job
   * The running jobs info is stored in manager YAML
   * @return false if running jobs exists this name
   */
  def checkManagerYaml(): Boolean = {
    val yamlParser = new Yaml()

    try {
      new FileInputStream(new File(Conventions.TMP_MANAGER_YAML))
    } catch {
      case _ => new File(Conventions.TMP_MANAGER_YAML).createNewFile()
    }
    val input = new FileInputStream(new File(Conventions.TMP_MANAGER_YAML))
    val loaded = yamlParser.load(input)
      .asInstanceOf[LinkedHashMap[String, util.LinkedHashMap[String, String]]]
    val configList = if (loaded != null) {
      loaded
    } else {
      new LinkedHashMap[String, util.LinkedHashMap[String, String]]()
    }
    configList.asScala.foreach(m => {
      if (m._2.get("name") == jobName) {
        return false
      }
    })
    true
  }

  /**
   * Add or remove job info in manager YAML,
   * manager YAML stores the info of running Cluster Serving jobs
   * @param jobId the jobId of this job
   * @param remove the flag to control whether to add job to manager YAML
   *               or to remove the job in manager YAML
   */
  def updateManagerYaml(jobId: String, remove: Boolean = false): Unit = {
    println("Updating YAML of Cluster Serving Manager")
    val yamlParser = new Yaml()

    try {
      new FileInputStream(new File(Conventions.TMP_MANAGER_YAML))
    } catch {
      case _ => new File(Conventions.TMP_MANAGER_YAML).createNewFile()
    }
    val input = new FileInputStream(new File(Conventions.TMP_MANAGER_YAML))
    val loaded = yamlParser.load(input)
      .asInstanceOf[LinkedHashMap[String, util.LinkedHashMap[String, String]]]
    val configList = if (loaded != null) {
      loaded
    } else {
      new LinkedHashMap[String, util.LinkedHashMap[String, String]]()
    }


    if (remove) {
      var uuid = ""
      configList.asScala.foreach(m => {
        if (m._2.get("id") == jobId) {
          uuid = m._1
        }
      })
      configList.remove(uuid)
    } else {
      val newJob = new HM()
      newJob.put("name", jobName)
      newJob.put("id", jobId)
      println(s"Adding job $jobName to manager YAML")
      configList.put(UUID.randomUUID().toString, newJob)
    }

    val outputWriter = new FileWriter(Conventions.TMP_MANAGER_YAML)
    yamlParser.dump(configList, outputWriter)
  }

  /**
   * For dynamically update model, not used currently
   * @return
   */
  def updateConfig(): Boolean = {
    val lastModTime = Files.getLastModifiedTime(Paths.get(configPath)).toString
    if (this.lastModTime != lastModTime) {
      loadConfig()
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
      ClusterServing.logger.debug(s"Config list ${configList.toString} " +
        s"get key $key, value $configValue")
      configValue
    }
  }
  /**
   * Load inference model
   * The concurrent number of inference model depends on
   * backend engine type
   * @return
   */
  def loadInferenceModel(concurrentNum: Int = 0): InferenceModel = {
    if (modelDir != null) {
      parseModelType(modelDir)
    }
    if (modelType.startsWith("tensorflow")) {
      chwFlag = false
    }
    // Allow concurrent number overwrite
    if (concurrentNum > 0) {
      modelPar = concurrentNum
    }
    ClusterServing.logger.info(s"Cluster Serving load Inference Model with Parallelism $modelPar")
    val model = new InferenceModel(modelPar)

    // Used for Tensorflow Model, it could not have intraThreadNum > 2^8
    // in some models, thus intraThreadNum should be limited

    var secret: String = null
    var salt: String = null
    if (modelEncrypted) {
      val jedis = new Jedis(redisHost, redisPort)
      while (secret == null || salt == null) {
        secret = jedis.hget(Conventions.MODEL_SECURED_KEY, Conventions.MODEL_SECURED_SECRET)
        salt = jedis.hget(Conventions.MODEL_SECURED_KEY, Conventions.MODEL_SECURED_SALT)
        ClusterServing.logger.info("Waiting for Model Encrypted Secret and Salt in Redis," +
          "please put them in model_secured -> secret and " +
          "model_secured -> salt")
        ClusterServing.logger.info("Retrying in 3 seconds...")
        Thread.sleep(3000)
      }

    }
    modelType match {
      case "caffe" => model.doLoadCaffe(defPath, weightPath, blas = blasFlag)
      case "bigdl" => model.doLoadBigDL(weightPath, blas = blasFlag)
      case "tensorflowFrozenModel" =>
        model.doLoadTensorflow(weightPath, "frozenModel", 1, 1, true)
      case "tensorflowSavedModel" =>
        model.doLoadTensorflow(weightPath, "savedModel", null, null)
      case "pytorch" => model.doLoadPyTorch(weightPath)
      case "keras" => logError("Keras currently not supported in Cluster Serving," +
        "consider transform it to Tensorflow")
      case "openvino" => modelEncrypted match {
        case true => model.doLoadEncryptedOpenVINO(defPath, weightPath, secret, salt, coreNum)
        case false => model.doLoadOpenVINO(defPath, weightPath, coreNum)
      }
      case _ => logError("Invalid model type, please check your model directory")
    }
    model
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
    ClusterServing.logger.error(msg)
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

    if (fileList == null) {
      logError("Your model path provided in config is empty, please check your model path.")
    }
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
      if (modelType == null) logError("There is no model detected in your directory." +
        "Please refer to document for supported model types.")
    }
    else {
      modelType = modelType.toLowerCase
    }

  }

}
object ClusterServingHelper {
  /**
   * Method wrapped for external use only
   * @param modelDir directory of model
   * @param concurrentNumber model concurrent number
   * @return
   */
  def loadModelfromDir(modelDir: String, concurrentNumber: Int = 1): (InferenceModel, String) = {
    val helper = new ClusterServingHelper(_modelDir = modelDir)
    helper.parseModelType(modelDir)
    (helper.loadInferenceModel(concurrentNumber), helper.modelType)
  }
}
