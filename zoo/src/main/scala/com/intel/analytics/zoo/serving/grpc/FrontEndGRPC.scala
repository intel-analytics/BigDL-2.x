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

package com.intel.analytics.zoo.serving.grpc

import akka.actor.ActorSystem
import akka.http.scaladsl.model.{HttpRequest, HttpResponse}
import akka.http.scaladsl.Http
import com.intel.analytics.zoo.serving.utils.Conventions
import com.typesafe.config.ConfigFactory
import com.codahale.metrics.{MetricRegistry, Timer}
import com.intel.analytics.zoo.serving.http.{ClusterServingMetaData, ClusterServingServable, InferenceModelMetaData, InferenceModelServable, Instances, JsonUtil, Predictions, ServableLoadException, ServableManager, Supportive}

import scala.collection.mutable
import org.slf4j.LoggerFactory

import scala.collection.mutable.ListBuffer
import scala.concurrent.{ExecutionContext, Future}

object FrontEndGRPC extends Supportive{
  var servableManager : ServableManager = _
  def main(args: Array[String]): Unit = {
    val arguments = timing("parse arguments")() {
      argumentsParser.parse(args, FrontEndAppArguments()) match {
        case Some(arguments) => logger.info(s"starting with $arguments"); arguments
        case None => argumentsParser.failure("miss args, please see the usage info"); null
      }
    }
    val conf = ConfigFactory
      .parseString("akka.http.server.preview.enable-http2 = on")
      .withFallback(ConfigFactory.defaultApplication())
    val system = ActorSystem("FrontEndGRPC", conf)
    implicit val sys: ActorSystem = system
    implicit val ec: ExecutionContext = sys.dispatcher

    servableManager = new ServableManager
    logger.info("Multi Serving Mode")
    timing("load servable manager")() {
      try servableManager.load(arguments.servableManagerPath, purePredictTimersMap,
        modelInferenceTimersMap)
      catch {
        case e: ServableLoadException =>
          throw e
        case e =>
          val exampleYaml =
            """
                ---
                 modelMetaDataList:
                 - !<ClusterServingMetaData>
                    modelName: "1"
                    modelVersion:"1.0"
                    redisHost: "localhost"
                    redisPort: "6381"
                    redisInputQueue: "serving_stream2"
                    redisOutputQueue: "cluster-serving_serving_stream2:"
                 - !<InflerenceModelMetaData>
                    modelName: "1"
                    modelVersion:"1.0"
                    modelPath:"/"
                    modelType:"OpenVINO"
                    features:
                      - "a"
                      - "b"
              """
          logger.info("Example Format of Input:" + exampleYaml)
          throw e
      }
    }
    logger.info("Servable Manager Load Success!")


    // WARN: Firstly, only single connection is supported

    // Create service handlers
    val service: HttpRequest => Future[HttpResponse] =
      FrontEndGRPCServiceHandler(new FrontEndGRPCServiceImpl(arguments))

    // Bind service handler servers to localhost:8080/8081
    val binding = Http().newServerAt(arguments.interface, arguments.port).bind(service)

    // report successful binding
    binding.foreach { binding => println(s"gRPC server bound to: ${binding.localAddress}") }

    binding
    // ActorSystem threads will keep the app alive until `system.terminate()` is called
  }


  val argumentsParser = new scopt.OptionParser[FrontEndAppArguments]("AZ Serving") {
    head("Analytics Zoo Serving FrontEndGRPC.scala")
    opt[String]('i', "interface")
      .action((x, c) => c.copy(interface = x))
      .text("network interface of frontend")
    opt[Int]('p', "port")
      .action((x, c) => c.copy(port = x))
      .text("network port of frontend")
    opt[Int]('s', "securePort")
      .action((x, c) => c.copy(securePort = x))
      .text("https port of frontend")
    opt[String]('h', "redisHost")
      .action((x, c) => c.copy(redisHost = x))
      .text("host of redis")
    opt[Int]('r', "redisPort")
      .action((x, c) => c.copy(redisPort = x))
      .text("port of redis")
    opt[String]('i', "redisInputQueue")
      .action((x, c) => c.copy(redisInputQueue = x))
      .text("input queue of redis")
    opt[String]('o', "redisOutputQueue")
      .action((x, c) => c.copy(redisOutputQueue = x))
      .text("output queue  of redis")
    opt[Int]('l', "parallelism")
      .action((x, c) => c.copy(parallelism = x))
      .text("parallelism of frontend")
    opt[Int]('t', "timeWindow")
      .action((x, c) => c.copy(timeWindow = x))
      .text("timeWindow of frontend")
    opt[Int]('c', "countWindow")
      .action((x, c) => c.copy(countWindow = x))
      .text("countWindow of frontend")
    opt[Boolean]('e', "tokenBucketEnabled")
      .action((x, c) => c.copy(tokenBucketEnabled = x))
      .text("Token Bucket Enabled or not")
    opt[Int]('k', "tokensPerSecond")
      .action((x, c) => c.copy(tokensPerSecond = x))
      .text("tokens per second")
    opt[Int]('a', "tokenAcquireTimeout")
      .action((x, c) => c.copy(tokenAcquireTimeout = x))
      .text("token acquire timeout")
    opt[Boolean]('s', "httpsEnabled")
      .action((x, c) => c.copy(httpsEnabled = x))
      .text("https enabled or not")
    opt[String]('p', "httpsKeyStorePath")
      .action((x, c) => c.copy(httpsKeyStorePath = x))
      .text("https keyStore path")
    opt[String]('w', "httpsKeyStoreToken")
      .action((x, c) => c.copy(httpsKeyStoreToken = x))
      .text("https keyStore token")
    opt[Boolean]('s', "redisSecureEnabled")
      .action((x, c) => c.copy(redisSecureEnabled = x))
      .text("redis secure enabled or not")
    opt[Boolean]('s', "httpsEnabled")
      .action((x, c) => c.copy(httpsEnabled = x))
      .text("https enabled or not")
    opt[String]('p', "redissTrustStorePath")
      .action((x, c) => c.copy(redissTrustStorePath = x))
      .text("rediss trustStore path")
    opt[String]('w', "redissTrustStoreToken")
      .action((x, c) => c.copy(redissTrustStoreToken = x))
      .text("rediss trustStore password")
    opt[String]('z', "servableManagerConfPath")
      .action((x, c) => c.copy(servableManagerPath = x))
      .text("servableManagerConfPath")
  }


  class FrontEndGRPCServiceImpl(args: FrontEndAppArguments) extends FrontEndGRPCService {
    val logger = LoggerFactory.getLogger(getClass)

    override def ping(in: Empty): Future[StringReply] = {
      timing("ping")(overallRequestTimer) {
        println(s"welcome to analytics zoo grpc serving frontend")
        Future.successful(StringReply("welcome to analytics zoo grpc serving frontend"))
      }
    }

    override def getMetrics(in: Empty): Future[MetricsReply] = {
      println(s"welcome to analytics zoo grpc serving frontend")
      timing("metrics")(overallRequestTimer, metricsRequestTimer) {
        val keys = metrics.getTimers().keySet()
        val servingMetrics = keys.toArray.map(key => {
          val timer = metrics.getTimers().get(key)
          MetricsReply.Metric(key.toString, timer.getCount, timer.getMeanRate,
            timer.getSnapshot.getMin / 1000000, timer.getSnapshot.getMax / 1000000,
            timer.getSnapshot.getMean / 1000000, timer.getSnapshot.getMedian / 1000000,
            timer.getSnapshot.getStdDev / 1000000, timer.getSnapshot.get75thPercentile() / 1000000,
            timer.getSnapshot.get95thPercentile() / 1000000,
            timer.getSnapshot.get98thPercentile() / 1000000,
            timer.getSnapshot.get99thPercentile() / 1000000,
            timer.getSnapshot.get999thPercentile() / 1000000)
        }).toList
        Future.successful(MetricsReply(servingMetrics))
      }
    }

    override def getAllModels(in: Empty): Future[ModelsReply] = {
      timing("get All Models")(overallRequestTimer, servablesRetriveTimer) {
        try {
          val servables = servableManager.retriveAllServables
          var inferenceModelList = ListBuffer[InferenceModelGRPCMetaData]()
          var clusterServingList = ListBuffer[ClusterServingGRPCMetaData]()
          servables.foreach(e =>
            e.getMetaData match {
              case (metaData: InferenceModelMetaData) =>
                val inferenceModelGRPCMetaData = InferenceModelGRPCMetaData(metaData.modelName,
                  metaData.modelVersion, metaData.modelPath, metaData.modelType,
                  metaData.weightPath, metaData.modelConCurrentNum, metaData.inputCompileType,
                  metaData.features.mkString(","))
                inferenceModelList += inferenceModelGRPCMetaData
              case (metaData: ClusterServingMetaData) =>
                val clusterServingGRPCMetaData = ClusterServingGRPCMetaData(metaData.modelName,
                  metaData.modelVersion, metaData.redisHost, metaData.redisPort,
                  metaData.redisInputQueue, metaData.redisOutputQueue, metaData.timeWindow,
                  metaData.countWindow, metaData.redisSecureEnabled, metaData.redisTrustStorePath,
                  metaData.redisTrustStoreToken, metaData.features.mkString(","))
                clusterServingList += clusterServingGRPCMetaData
            })
          Future.successful(ModelsReply(inferenceModelList, clusterServingList))
        }
        catch {
          case e =>
            Future.failed(e)
        }
      }
    }

    override def getModelsWithName(in: GetModelsWithNameReq): Future[ModelsReply] = {
      timing("get Models With Name")(overallRequestTimer, servablesRetriveTimer) {
        try {
          val servables = servableManager.retriveServables(in.modelName)
          var inferenceModelList = ListBuffer[InferenceModelGRPCMetaData]()
          var clusterServingList = ListBuffer[ClusterServingGRPCMetaData]()
          servables.foreach(e =>
            e.getMetaData match {
              case (metaData: InferenceModelMetaData) =>
                val inferenceModelGRPCMetaData = InferenceModelGRPCMetaData(metaData.modelName,
                  metaData.modelVersion, metaData.modelPath, metaData.modelType,
                  metaData.weightPath, metaData.modelConCurrentNum, metaData.inputCompileType,
                  metaData.features.mkString(","))
                inferenceModelList += inferenceModelGRPCMetaData
              case (metaData: ClusterServingMetaData) =>
                val clusterServingGRPCMetaData = ClusterServingGRPCMetaData(metaData.modelName,
                  metaData.modelVersion, metaData.redisHost, metaData.redisPort,
                  metaData.redisInputQueue, metaData.redisOutputQueue, metaData.timeWindow,
                  metaData.countWindow, metaData.redisSecureEnabled, metaData.redisTrustStorePath,
                  metaData.redisTrustStoreToken, metaData.features.mkString(","))
                clusterServingList += clusterServingGRPCMetaData
            })
          Future.successful(ModelsReply(inferenceModelList, clusterServingList))
        }
        catch {
          case e =>
            Future.failed(e)
        }
      }
    }

    override def getModelsWithNameAndVersion(in: GetModelsWithNameAndVersionReq):
       Future[ModelsReply] = {
      timing("get Model With Name And Version")(overallRequestTimer, servableRetriveTimer) {
        try {
          val servable = servableManager.retriveServable(in.modelName, in.modelVersion)
          servable.getMetaData match {
            case (metaData: InferenceModelMetaData) =>
              val inferenceModelList =
                List[InferenceModelGRPCMetaData](InferenceModelGRPCMetaData(metaData.modelName,
                  metaData.modelVersion, metaData.modelPath, metaData.modelType,
                  metaData.weightPath, metaData.modelConCurrentNum, metaData.inputCompileType,
                  metaData.features.mkString(",")))
              val clusterServingList = List[ClusterServingGRPCMetaData]()
              Future.successful(ModelsReply(inferenceModelList, clusterServingList))
            case (metaData: ClusterServingMetaData) =>
              val inferenceModelList = List[InferenceModelGRPCMetaData]()
              val clusterServingList =
                List[ClusterServingGRPCMetaData](ClusterServingGRPCMetaData(metaData.modelName,
                  metaData.modelVersion, metaData.redisHost, metaData.redisPort,
                  metaData.redisInputQueue, metaData.redisOutputQueue, metaData.timeWindow,
                  metaData.countWindow, metaData.redisSecureEnabled, metaData.redisTrustStorePath,
                  metaData.redisTrustStoreToken, metaData.features.mkString(",")))
              Future.successful(ModelsReply(inferenceModelList, clusterServingList))
          }
        }
        catch {
          case e =>
            Future.failed(e)
        }
      }
    }

    override def predict(in: PredictReq): Future[PredictReply] = {
      timing("backend inference")(overallRequestTimer, backendInferenceTimer) {
          try {
            logger.info("model name: " + in.modelName + ", model version: " + in.modelVersion)
            val servable = timing("servable retrive")(servableRetriveTimer) {
              servableManager.retriveServable(in.modelName, in.modelVersion)
            }
            val modelInferenceTimer = modelInferenceTimersMap(in.modelName)(in.modelVersion)
            servable match {
              case _: ClusterServingServable =>
                val result = timing("cluster serving inference")(predictRequestTimer) {
                  val instances = timing("json deserialization")() {
                    JsonUtil.fromJson(classOf[Instances], in.input)
                  }
                  val outputs = timing("model inference")(modelInferenceTimer) {
                    servable.predict(instances)
                  }
                  Predictions(outputs)
                }
                timing("cluster serving response complete")() {
                  Future.successful(PredictReply(result.toString))
                }
              case _: InferenceModelServable =>
                val result = timing("inference model inference")(predictRequestTimer) {
                  val outputs = servable.getMetaData.
                    asInstanceOf[InferenceModelMetaData].inputCompileType match {
                    case "direct" => timing("model inference")(modelInferenceTimer) {
                      servable.predict(in.input)
                    }
                    case "instance" =>
                      val instances = timing("json deserialization")() {
                        JsonUtil.fromJson(classOf[Instances], in.input)
                      }
                      timing("model inference")(modelInferenceTimer) {
                        servable.predict(instances)
                      }
                  }
                  JsonUtil.toJson(outputs.map(_.result))
                }
                timing("inference model response complete")() {
                  Future.successful(PredictReply(result.toString))
                }
            }
          }
          catch {
            case e =>
              Future.failed(e)
          }
        }
      }
  }

  logger.info(s"metrics init")
  val metrics = new MetricRegistry
  val overallRequestTimer = metrics.timer("zoo.serving.request.overall")
  val predictRequestTimer = metrics.timer("zoo.serving.request.predict")
  val servableRetriveTimer = metrics.timer("zoo.serving.retrive.servable")
  val servablesRetriveTimer = metrics.timer("zoo.serving.retrive.servables")
  val backendInferenceTimer = metrics.timer("zoo.serving.backend.inference")
  val putRedisTimer = metrics.timer("zoo.serving.redis.put")
  val getRedisTimer = metrics.timer("zoo.serving.redis.get")
  val waitRedisTimer = metrics.timer("zoo.serving.redis.wait")
  val metricsRequestTimer = metrics.timer("zoo.serving.request.metrics")
  val modelInferenceTimersMap = new mutable.HashMap[String, mutable.HashMap[String, Timer]]
  val purePredictTimersMap = new mutable.HashMap[String, mutable.HashMap[String, Timer]]
  val makeActivityTimer = metrics.timer("zoo.serving.activity.make")
  val handleResponseTimer = metrics.timer("zoo.serving.response.handling")
}



case class FrontEndAppArguments(
                                 interface: String = "127.0.0.1",
                                 port: Int = 8080,
                                 securePort: Int = 10023,
                                 redisHost: String = "localhost",
                                 redisPort: Int = 6379,
                                 redisInputQueue: String = Conventions.SERVING_STREAM_DEFAULT_NAME,
                                 redisOutputQueue: String =
                                 Conventions.RESULT_PREFIX + Conventions.SERVING_STREAM_DEFAULT_NAME
                                   + ":",
                                 parallelism: Int = 1000,
                                 timeWindow: Int = 0,
                                 countWindow: Int = 0,
                                 tokenBucketEnabled: Boolean = false,
                                 tokensPerSecond: Int = 100,
                                 tokenAcquireTimeout: Int = 100,
                                 httpsEnabled: Boolean = false,
                                 httpsKeyStorePath: String = null,
                                 httpsKeyStoreToken: String = "1234qwer",
                                 redisSecureEnabled: Boolean = false,
                                 redissTrustStorePath: String = null,
                                 redissTrustStoreToken: String = "1234qwer",
                                 servableManagerPath: String = "./servables-conf.yaml"
                               )