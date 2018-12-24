package com.intel.analytics.zoo.examples.recommendation

import com.intel.analytics.bigdl.dataset.{MiniBatch, Sample}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.visualization.{TrainSummary, ValidationSummary}
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.pipeline.api.keras.layers.{Dense, Embedding, GRU}
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.objectives.SparseCategoricalCrossEntropy
import com.intel.analytics.zoo.models.recommendation.SessionRecommender
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql.{Column, DataFrame, SQLContext}
import org.apache.spark.sql.functions._
import org.apache.spark.storage.StorageLevel
import scopt.OptionParser

import scala.collection.mutable

case class SessionParams(val nEpochs: Int = 2,
                         val batchSize: Int = 4000,
                         val inputDir: String = "/Users/guoqiong/intelWork/projects/sessionRec/yoochoose-data/yoochoose-test.dat",
                         val logDir: String = "./log/",
                         val featureLength:Int = 10
                      )

case class Session(session: String, timestamp: String, item: String, category: String)

object SessionRecExp {

  def main(args: Array[String]): Unit = {

    val defaultParams = SessionParams()

    val parser = new OptionParser[SessionParams]("NCF Example") {
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

  def run(params: SessionParams): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = new SparkConf()
    conf.setAppName("SessionRecExample").set("spark.sql.crossJoin.enabled", "true")
    val sc = NNContext.initNNContext(conf)
    val sqlContext = SQLContext.getOrCreate(sc)

    val (sessionDF, sessionCnt, itemCnt) = loadPublicData(sqlContext, params.inputDir)
    println("records:" + sessionDF.count())
    println("sessionCnt:" + sessionCnt)
    println("itemCnt:" + itemCnt)

    val featureRdds = assemblyFeature(sessionDF.sample(false, 0.1), params.featureLength)
    val Array(train, validation) = featureRdds.randomSplit(Array(0.8, 0.2))


    val model = SessionRecommender[Float](itemCnt, 20, params.featureLength, 100)
  //  val model = buildModel(itemCnt.toInt, params.maxLength, params.embedOutDim)

    val optimizer: Optimizer[Float, MiniBatch[Float]] = Optimizer(
      model = model,
      sampleRDD = train,
      criterion = new SparseCategoricalCrossEntropy[Float](logProbAsInput = true, zeroBasedLabel = false),
      batchSize = params.batchSize
    )

    val trainSummary = TrainSummary(logDir = "./log", appName = "recommenderOD")
    trainSummary.setSummaryTrigger("Loss", trigger = Trigger.severalIteration(1))
    val valSummary = ValidationSummary(logDir = "./log", appName = "recommenderOD")

    optimizer
      .setOptimMethod(new RMSprop[Float]())
      .setValidation(Trigger.everyEpoch, validation, Array(new Top5Accuracy[Float]()), params.batchSize)
      .setEndWhen(Trigger.maxEpoch(params.nEpochs))
      .setTrainSummary(trainSummary)
      .setValidationSummary(valSummary)

    val trained_model = optimizer.optimize()

    trained_model.saveModule(params.inputDir, null, overWrite = true)
    println("Model has been saved")
  }

  def buildModel(itemCnt: Int,
                 maxLength: Int,
                 embedOutDim: Int
                ): Sequential[Float] = {
    val model = Sequential[Float]()

    model.add(Embedding[Float](itemCnt, embedOutDim, init = "normal", inputLength = maxLength))
      //  .add(GRU[Float](40, returnSequences = true))
      .add(GRU[Float](200, returnSequences = false))
      .add(Dense[Float](itemCnt, activation = "log_softmax"))
    model
  }

  def loadPublicData(sqlContext: SQLContext, dataPath: String) = {
    import sqlContext.implicits._
    val sessions = sqlContext.read.text(dataPath).as[String]
      .map(x => {
        val line = x.split(",")
        Session(line(0), line(1), line(2), line(3))
      }).toDF()
      .select("session", "item")
      .filter(col("session").isNotNull && col("item").isNotNull)

    val dataIndex1 = new StringIndexer().setInputCol("session").setOutputCol("sessionId")
    val dataIndex2 = new StringIndexer().setInputCol("item").setOutputCol("itemId")
    val pipeline = new Pipeline()
      .setStages(Array(dataIndex1, dataIndex2))
    val model = pipeline.fit(sessions)
    val sessionsIndexed = model.transform(sessions)
      .withColumn("sessionId", col("sessionId") + 1)
      .withColumn("itemId", col("itemId") + 1)
      .drop("session", "item")

    val minMaxRow = sessionsIndexed.agg(max("sessionId"), max("itemId")).collect()(0)

    val sessionsOut = sessionsIndexed
      .groupBy("sessionId")
      .agg(collect_list("itemId").alias("itemIds"))
      .filter(col("itemIds").isNotNull && size(col("itemIds")) > 1)

    (sessionsOut, minMaxRow.getDouble(0), minMaxRow.getDouble(1))
  }

  def assemblyFeature(sessions: DataFrame, maxLength: Int) = {

    /*PrePad UDF*/
    def getFeatures: mutable.WrappedArray[java.lang.Double] => Array[Float] = x => {
      val padded = if (x.length > maxLength) x.array.map(_.toFloat)
      else Array.fill[Float](maxLength - x.length + 1)(0) ++ x.array.map(_.toFloat)
      padded.takeRight(maxLength + 1).dropRight(1)
    }

    val getFeaturesUDF = udf(getFeatures)

    /*Get label UDF*/
    def getLabel: mutable.WrappedArray[java.lang.Double] => Float = x => {
      x.takeRight(1).head.floatValue() + 1
    }

    val getLabelUDF = udf(getLabel)

    val featuresDF = sessions
      .withColumn("features", getFeaturesUDF(col("itemIds")))
      .withColumn("label", getLabelUDF(col("itemIds")))

    /*DataFrame to sample*/
    val rddOfSample = featuresDF.rdd.map(r => {
      val label = Tensor[Float](T(r.getAs[Float]("label")))
      val featureArray = r.getAs[mutable.WrappedArray[java.lang.Float]]("features").array.map(_.toFloat)
      val feature = Tensor(featureArray, Array(maxLength))
      Sample(feature, label)
    })

    rddOfSample
  }

}
