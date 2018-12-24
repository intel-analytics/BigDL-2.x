package com.intel.analytics.zoo.examples.recommendation

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dataset.{MiniBatch, Sample}
import com.intel.analytics.bigdl.nn.ClassNLLCriterion
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.visualization.{TrainSummary, ValidationSummary}
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.models.recommendation.{KerasRNN, NeuralCF, UserItemFeature, Utils}
import com.intel.analytics.zoo.pipeline.api.keras.layers.{Dense, Embedding, GRU}
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.objectives.SparseCategoricalCrossEntropy
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.sql.functions._
import org.apache.spark.storage.StorageLevel
import scopt.OptionParser

import scala.collection.mutable

case class ModelParams(
                        maxLength:Int = 8,
                        maxEpoch:Int = 2,
                        batchSize:Int = 4000,
                        embedOutDim:Int = 20,
                        inputDir:String = "/Users/guoqiong/intelWork/projects/sessionRec/yoochoose-data/yoochoose-clicks.dat",
                        logDir:String = "./log/",
                        rnnName:String = "rnnModel"
                      )


case class Session(sessionId: String, itemId: String)

object SessionRecExp {

  def main(args: Array[String]): Unit = {

    val defaultParams = ModelParams()

    val parser = new OptionParser[ModelParams]("NCF Example") {
      opt[String]("inputDir")
        .text(s"inputDir")
        .action((x, c) => c.copy(inputDir = x))
      opt[Int]('b', "batchSize")
        .text(s"batchSize")
        .action((x, c) => c.copy(batchSize = x.toInt))
      opt[Int]('e', "maxEpoch")
        .text("epoch numbers")
        .action((x, c) => c.copy(maxEpoch = x))

    }

    parser.parse(args, defaultParams).map {
      params =>
        run(params)
    } getOrElse {
      System.exit(1)
    }
  }

  def run(params: ModelParams): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = new SparkConf()
    conf.setAppName("SessionRecExample").set("spark.sql.crossJoin.enabled", "true")
    val sc = NNContext.initNNContext(conf)
    val sqlContext = SQLContext.getOrCreate(sc)

    val sessionDF = loadPublicData(sqlContext, params.inputDir)//.filter("sessionId <= 10000")

    val itemCount = sessionDF.select("itemId").distinct().count()
    println("items:" + itemCount)

    sessionDF.printSchema()
    sessionDF.show()

    val dataIndex1 = new StringIndexer().setInputCol("sessionId").setOutputCol("sessionIdIndex")
    val dataIndex2 = new StringIndexer().setInputCol("itemId").setOutputCol("itemIdIndex")

    val pipeline = new Pipeline()
      .setStages(Array(dataIndex1, dataIndex2))

    val model = pipeline.fit(sessionDF)
    val data1 = model.transform(sessionDF)
      .withColumn("sessionIdIndex",col("sessionIdIndex") +1)
      .withColumn("itemIdIndex",col("itemIdIndex") +1)
      .persist(StorageLevel.DISK_ONLY)


    data1.show()
    data1.printSchema()

    /*Collect item to sequence*/
    val data2 = data1.groupBy("sessionIdIndex")
      .agg(collect_list("itemIdIndex").alias("items"))
      .filter(col("items").isNotNull).persist(StorageLevel.DISK_ONLY)

    data2.printSchema()
    data2.show()


    /*PrePad UDF*/
    def prePadding: mutable.WrappedArray[java.lang.Double] => Array[Float] = x => {
      val item = if (x.length > params.maxLength) x.array.map(_.toFloat)
      else Array.fill[Float](params.maxLength - x.length + 1)(0) ++ x.array.map(_.toFloat)
      val item2 = item.takeRight(params.maxLength + 1)
      val item3 = item2.dropRight(1)
      item3
    }

    val prePaddingUDF = udf(prePadding)

    /*Get label UDF*/
    def getLabel: mutable.WrappedArray[java.lang.Double] => Float = x => {
      x.takeRight(1).head.floatValue() + 1
    }
    val getLabelUDF = udf(getLabel)

    val data3 = data2
      .withColumn("features", prePaddingUDF(col("items")))
      .withColumn("label", getLabelUDF(col("items")))
      .persist(StorageLevel.DISK_ONLY)

    data3.show(false)
    data3.printSchema()


    val outSize = data3.rdd.map(_.getAs[Float]("label")).max.toInt
    println(outSize)

    /*DataFrame to sample*/
    val trainSample = data3.rdd.map(r => {
      val label = Tensor[Float](T(r.getAs[Float]("label")))
      val array = r.getAs[mutable.WrappedArray[java.lang.Float]]("features").array.map(_.toFloat)
      val vec = Tensor(array, Array(params.maxLength))
      Sample(vec, label)
    })

    println("Sample feature print: \n"+ trainSample.take(1).head.feature())
    println("Sample label print: \n" + trainSample.take(1).head.label())

    /*Train rnn model using Keras API*/
    val model1 = buildModel(outSize, itemCount.toInt, params.maxLength, params.embedOutDim)
    train(model1, trainSample, params.inputDir, params.rnnName, params.logDir, params.maxEpoch, params.batchSize)



  }

  def buildModel(
                  numClasses: Int,
                  skuCount: Int,
                  maxLength: Int,
                  embedOutDim: Int
                ): Sequential[Float] = {
    val model = Sequential[Float]()

    model.add(Embedding[Float](skuCount + 1, embedOutDim, init = "normal", inputLength = maxLength))
      //  .add(GRU[Float](40, returnSequences = true))
      .add(GRU[Float](200, returnSequences = false))
      .add(Dense[Float](numClasses, activation = "log_softmax"))
    model
  }

  def train(
             model: Sequential[Float],
             train: RDD[Sample[Float]],
             inputDir: String,
             rnnName: String,
             logDir: String,
             maxEpoch: Int,
             batchSize: Int
           ): Module[Float] = {

    val split = train.randomSplit(Array(0.8, 0.2), 100)
    val trainRDD = split(0)
    val testRDD = split(1)

    trainRDD.cache()
    testRDD.cache()

    println(model.summary())
    println("trainingrdd" + trainRDD.count())

    val optimizer: Optimizer[Float, MiniBatch[Float]] = Optimizer(
      model = model,
      sampleRDD = trainRDD,
      criterion = new SparseCategoricalCrossEntropy[Float](logProbAsInput = true, zeroBasedLabel = false),
      batchSize = batchSize
    )

    val trainSummary = TrainSummary(logDir = "./log", appName = "recommenderOD")
    trainSummary.setSummaryTrigger("Loss", trigger = Trigger.severalIteration(1))
    val valSummary = ValidationSummary(logDir = "./log", appName = "recommenderOD")

    optimizer
      .setOptimMethod(new RMSprop[Float]())
      .setValidation(Trigger.everyEpoch, testRDD, Array(new Top5Accuracy[Float]()), batchSize)
      .setEndWhen(Trigger.maxEpoch(maxEpoch))
      .setTrainSummary(trainSummary)
      .setValidationSummary(valSummary)

    val trained_model = optimizer.optimize()

    trained_model.saveModule(inputDir + rnnName, null, overWrite = true)
    println("Model has been saved")

    trained_model
  }
  def loadPublicData(sqlContext: SQLContext, dataPath: String): DataFrame = {
    import sqlContext.implicits._
    val sessions = sqlContext.read.text(dataPath).as[String]
      .map(x => {
        val data = x.split(",")
        Session(data(0), data(2))
      }).toDF()

    sessions
  }

  def assemblyFeature(isImplicit: Boolean = false,
                      indexed: DataFrame,
                      userCount: Int,
                      itemCount: Int): RDD[UserItemFeature[Float]] = {

    val unioned = if (isImplicit) {
      val negativeDF = Utils.getNegativeSamples(indexed)
      negativeDF.unionAll(indexed.withColumn("label", lit(2)))
    }
    else indexed

    val rddOfSample: RDD[UserItemFeature[Float]] = unioned
      .select("userId", "itemId", "label")
      .rdd.map(row => {
      val uid = row.getAs[Int](0)
      val iid = row.getAs[Int](1)

      val label = row.getAs[Int](2)
      val feature: Tensor[Float] = Tensor[Float](T(uid.toFloat, iid.toFloat))

      UserItemFeature(uid, iid, Sample(feature, Tensor[Float](T(label))))
    })
    rddOfSample
  }

}
