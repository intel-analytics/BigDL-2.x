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


import com.intel.analytics.zoo.serving.utils.Conventions

import com.codahale.metrics.{MetricRegistry, Timer}
import com.intel.analytics.zoo.grpc.ZooGrpcServer
import com.intel.analytics.zoo.serving.grpc.service.generated.FrontEndGRPCServiceGrpc.FrontEndGRPCServiceImplBase
import com.intel.analytics.zoo.serving.http.{ClusterServingMetaData, ClusterServingServable, InferenceModelMetaData, InferenceModelServable, Instances, JsonUtil, Predictions, ServableLoadException, ServableManager, Supportive}
import com.intel.analytics.zoo.serving.grpc.service.generated._
import io.grpc.stub.StreamObserver

import scala.collection.mutable
import scala.concurrent.Future

object FrontEndGRPCServiceImpl extends Supportive{
  def main(args: Array[String]): Unit = {
    val server = new ZooGrpcServer(new FrontEndGRPCServiceImpl(args))
    server.start()
    server.blockUntilShutdown()
  }
}

class FrontEndGRPCServiceImpl (args: Array[String]) extends FrontEndGRPCServiceImplBase
  with Supportive {
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

  val argumentsParser = new scopt.OptionParser[FrontEndAppArguments]("AZ Serving") {
    head("Analytics Zoo Serving FrontEndGRPCServiceImpl.scala")
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
  val arguments = timing("parse arguments")() {
    argumentsParser.parse(args, FrontEndAppArguments()) match {
      case Some(arguments) => logger.info(s"starting with $arguments"); arguments
      case None => argumentsParser.failure("miss args, please see the usage info"); null
    }
  }
  val servableManager = new ServableManager
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


  override def ping(in: Empty, responseObserver: StreamObserver[StringReply]):
  Unit = {
    timing("ping")(overallRequestTimer) {
      println(s"welcome to analytics zoo grpc serving frontend")
      val reply = StringReply.newBuilder.setMessage("welcome to analytics zoo " +
        "grpc serving frontend").build
      responseObserver.onNext(reply)
      responseObserver.onCompleted()
    }
  }

  override def getMetrics(in: Empty, responseObserver: StreamObserver[MetricsReply]):
  Unit = {
    println(s"welcome to analytics zoo grpc serving frontend")
    timing("metrics")(overallRequestTimer, metricsRequestTimer) {
      val keys = metrics.getTimers().keySet()
      val reply = MetricsReply.newBuilder()
      val servingMetrics = keys.toArray.foreach(key => {
        val timer = metrics.getTimers().get(key)
        val metric = MetricsReply.Metric.newBuilder
        metric.setName(key.toString)
        metric.setCount(timer.getCount)
        metric.setMeanRate(timer.getMeanRate)
        metric.setMin(timer.getSnapshot.getMin / 1000000)
        metric.setMax(timer.getSnapshot.getMax / 1000000)
        metric.setMean(timer.getSnapshot.getMean / 1000000)
        metric.setMedian(timer.getSnapshot.getMedian / 1000000)
        metric.setStdDev(timer.getSnapshot.getStdDev / 1000000)
        metric.setPercentile75Th(timer.getSnapshot.get75thPercentile() / 1000000)
        metric.setPercentile95Th(timer.getSnapshot.get95thPercentile() / 1000000)
        metric.setPercentile98Th(timer.getSnapshot.get98thPercentile() / 1000000)
        metric.setPercentile99Th(timer.getSnapshot.get99thPercentile() / 1000000)
        metric.setPercentile999Th(timer.getSnapshot.get999thPercentile() / 1000000)
        metric.build()
        reply.addMetrics(metric)
      })
      responseObserver.onNext(reply.build())
      responseObserver.onCompleted()
    }
  }

  override def getAllModels(in: Empty, responseObserver: StreamObserver[ModelsReply]):
  Unit = {
    timing("get All Models")(overallRequestTimer, servablesRetriveTimer) {
      try {
        val servables = servableManager.retriveAllServables
        val reply = ModelsReply.newBuilder()
        servables.foreach(e =>
          e.getMetaData match {
            case (metaData: InferenceModelMetaData) =>
              val inferenceModelGRPCMetaData = InferenceModelGRPCMetaData.newBuilder
              inferenceModelGRPCMetaData.setModelName(metaData.modelName)
              inferenceModelGRPCMetaData.setModelVersion(metaData.modelVersion)
              inferenceModelGRPCMetaData.setModelPath(metaData.modelPath)
              inferenceModelGRPCMetaData.setModelType(metaData.modelType)
              inferenceModelGRPCMetaData.setWeightPath(metaData.weightPath)
              inferenceModelGRPCMetaData.setModelConCurrentNum(metaData.modelConCurrentNum)
              inferenceModelGRPCMetaData.setInputCompileType(metaData.inputCompileType)
              inferenceModelGRPCMetaData.setFeatures(metaData.features.mkString(","))
              reply.addInferenceModelMetaDatas(inferenceModelGRPCMetaData)
            case (metaData: ClusterServingMetaData) =>
              val clusterServingGRPCMetaData = ClusterServingGRPCMetaData.newBuilder
              clusterServingGRPCMetaData.setModelName(metaData.modelName)
              clusterServingGRPCMetaData.setModelVersion(metaData.modelVersion)
              clusterServingGRPCMetaData.setRedisHost(metaData.redisHost)
              clusterServingGRPCMetaData.setRedisPort(metaData.redisPort)
              clusterServingGRPCMetaData.setRedisInputQueue(metaData.redisInputQueue)
              clusterServingGRPCMetaData.setRedisOutputQueue(metaData.redisOutputQueue)
              clusterServingGRPCMetaData.setTimeWindow(metaData.timeWindow)
              clusterServingGRPCMetaData.setCountWindow(metaData.countWindow)
              clusterServingGRPCMetaData.setRedisSecureEnabled(metaData.redisSecureEnabled)
              clusterServingGRPCMetaData.setRedisTrustStorePath(metaData.redisTrustStorePath)
              clusterServingGRPCMetaData.setRedisTrustStoreToken(metaData.redisTrustStoreToken)
              clusterServingGRPCMetaData.setFeatures(metaData.features.mkString(","))
              reply.addClusterServingMetaDatas(clusterServingGRPCMetaData)
          })
        responseObserver.onNext(reply.build())
        responseObserver.onCompleted()
      }
      catch {
        case e =>
          responseObserver.onError(e)
      }
    }
  }

  override def getModelsWithName(in: GetModelsWithNameReq,
                                 responseObserver: StreamObserver[ModelsReply]):
  Unit = {
    timing("get Models With Name")(overallRequestTimer, servablesRetriveTimer) {
      try {
        val servables = servableManager.retriveServables(in.getModelName)
        val reply = ModelsReply.newBuilder()
        servables.foreach(e =>
          e.getMetaData match {
            case (metaData: InferenceModelMetaData) =>
              val inferenceModelGRPCMetaData = InferenceModelGRPCMetaData.newBuilder
              inferenceModelGRPCMetaData.setModelName(metaData.modelName)
              inferenceModelGRPCMetaData.setModelVersion(metaData.modelVersion)
              inferenceModelGRPCMetaData.setModelPath(metaData.modelPath)
              inferenceModelGRPCMetaData.setModelType(metaData.modelType)
              inferenceModelGRPCMetaData.setWeightPath(metaData.weightPath)
              inferenceModelGRPCMetaData.setModelConCurrentNum(metaData.modelConCurrentNum)
              inferenceModelGRPCMetaData.setInputCompileType(metaData.inputCompileType)
              inferenceModelGRPCMetaData.setFeatures(metaData.features.mkString(","))
              reply.addInferenceModelMetaDatas(inferenceModelGRPCMetaData)
            case (metaData: ClusterServingMetaData) =>
              val clusterServingGRPCMetaData = ClusterServingGRPCMetaData.newBuilder
              clusterServingGRPCMetaData.setModelName(metaData.modelName)
              clusterServingGRPCMetaData.setModelVersion(metaData.modelVersion)
              clusterServingGRPCMetaData.setRedisHost(metaData.redisHost)
              clusterServingGRPCMetaData.setRedisPort(metaData.redisPort)
              clusterServingGRPCMetaData.setRedisInputQueue(metaData.redisInputQueue)
              clusterServingGRPCMetaData.setRedisOutputQueue(metaData.redisOutputQueue)
              clusterServingGRPCMetaData.setTimeWindow(metaData.timeWindow)
              clusterServingGRPCMetaData.setCountWindow(metaData.countWindow)
              clusterServingGRPCMetaData.setRedisSecureEnabled(metaData.redisSecureEnabled)
              clusterServingGRPCMetaData.setRedisTrustStorePath(metaData.redisTrustStorePath)
              clusterServingGRPCMetaData.setRedisTrustStoreToken(metaData.redisTrustStoreToken)
              clusterServingGRPCMetaData.setFeatures(metaData.features.mkString(","))
              reply.addClusterServingMetaDatas(clusterServingGRPCMetaData)
          })
        responseObserver.onNext(reply.build())
        responseObserver.onCompleted()
      }
      catch {
        case e =>
          responseObserver.onError(e)
      }
    }
  }

  override def getModelsWithNameAndVersion(in: GetModelsWithNameAndVersionReq,
                                           responseObserver: StreamObserver[ModelsReply]):
  Unit = {
    timing("get Model With Name And Version")(overallRequestTimer, servableRetriveTimer) {
      try {
        val servable = servableManager.retriveServable(in.getModelName, in.getModelVersion)
        val reply = ModelsReply.newBuilder()
        servable.getMetaData match {
          case (metaData: InferenceModelMetaData) =>
            val inferenceModelGRPCMetaData = InferenceModelGRPCMetaData.newBuilder
            inferenceModelGRPCMetaData.setModelName(metaData.modelName)
            inferenceModelGRPCMetaData.setModelVersion(metaData.modelVersion)
            inferenceModelGRPCMetaData.setModelPath(metaData.modelPath)
            inferenceModelGRPCMetaData.setModelType(metaData.modelType)
            inferenceModelGRPCMetaData.setWeightPath(metaData.weightPath)
            inferenceModelGRPCMetaData.setModelConCurrentNum(metaData.modelConCurrentNum)
            inferenceModelGRPCMetaData.setInputCompileType(metaData.inputCompileType)
            inferenceModelGRPCMetaData.setFeatures(metaData.features.mkString(","))
            reply.addInferenceModelMetaDatas(inferenceModelGRPCMetaData)
          case (metaData: ClusterServingMetaData) =>
            val clusterServingGRPCMetaData = ClusterServingGRPCMetaData.newBuilder
            clusterServingGRPCMetaData.setModelName(metaData.modelName)
            clusterServingGRPCMetaData.setModelVersion(metaData.modelVersion)
            clusterServingGRPCMetaData.setRedisHost(metaData.redisHost)
            clusterServingGRPCMetaData.setRedisPort(metaData.redisPort)
            clusterServingGRPCMetaData.setRedisInputQueue(metaData.redisInputQueue)
            clusterServingGRPCMetaData.setRedisOutputQueue(metaData.redisOutputQueue)
            clusterServingGRPCMetaData.setTimeWindow(metaData.timeWindow)
            clusterServingGRPCMetaData.setCountWindow(metaData.countWindow)
            clusterServingGRPCMetaData.setRedisSecureEnabled(metaData.redisSecureEnabled)
            clusterServingGRPCMetaData.setRedisTrustStorePath(metaData.redisTrustStorePath)
            clusterServingGRPCMetaData.setRedisTrustStoreToken(metaData.redisTrustStoreToken)
            clusterServingGRPCMetaData.setFeatures(metaData.features.mkString(","))
            reply.addClusterServingMetaDatas(clusterServingGRPCMetaData)
        }
        responseObserver.onNext(reply.build())
        responseObserver.onCompleted()
      }
      catch {
        case e =>
          responseObserver.onError(e)
      }
    }
  }

  override def predict(in: PredictReq,
                       responseObserver: StreamObserver[PredictReply]):
  Unit = {
    timing("backend inference")(overallRequestTimer, backendInferenceTimer) {
      try {
        logger.info("model name: " + in.getModelName + ", model version: " + in.getModelVersion)
        val servable = timing("servable retrive")(servableRetriveTimer) {
          servableManager.retriveServable(in.getModelName, in.getModelVersion)
        }
        val modelInferenceTimer = modelInferenceTimersMap(in.getModelName)(in.getModelVersion)
        val reply = servable match {
          case _: ClusterServingServable =>
            val result = timing("cluster serving inference")(predictRequestTimer) {
              val outputs = servable.getMetaData.
                asInstanceOf[ClusterServingMetaData].inputCompileType match {
                case "direct" =>
                  timing("model inference direct")(modelInferenceTimer) {
                    servable.predict(in.getInput)
                  }
                case "instance" =>
                  val instances = timing("json deserialization")() {
                    JsonUtil.fromJson(classOf[Instances], in.getInput)
                  }
                  timing("model inference")(modelInferenceTimer) {
                    servable.predict(instances)
                  }
              }
              JsonUtil.toJson(outputs.map(_.result))
            }
            timing("cluster serving response complete")() {
              PredictReply.newBuilder().setResponse(result.toString).build()
            }
          case _: InferenceModelServable =>
            val result = timing("inference model inference")(predictRequestTimer) {
              val outputs = servable.getMetaData.
                asInstanceOf[InferenceModelMetaData].inputCompileType match {
                case "direct" =>
                  timing("model inference")(modelInferenceTimer) {
                    servable.predict(in.getInput)
                  }
                case "instance" =>
                  val instances = timing("json deserialization")() {
                    JsonUtil.fromJson(classOf[Instances], in.getInput)
                  }
                  timing("model inference")(modelInferenceTimer) {
                    servable.predict(instances)
                  }
              }
              JsonUtil.toJson(outputs.map(_.result))
            }
            PredictReply.newBuilder().setResponse(result.toString).build()
        }
        responseObserver.onNext(reply)
        responseObserver.onCompleted()
      }
      catch {
        case e =>
          Future.failed(e)
      }
    }
  }

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
