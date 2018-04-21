package com.intel.analytics.zoo.examples.recommendation

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.nn.ClassNLLCriterion
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.{Adam, Optimizer, Trigger}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Engine, T}
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.models.recommendation.{NeuralCF, UserItemFeature, Utils}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, FloatType}
import scopt.OptionParser

case class NCFParams(val inputDir: String = "./data/ml-1m",
                     val batchSize: Int = 1000,
                     val nEpochs: Int = 10,
                     val learningRate: Double = 1e-3,
                     val learningRateDecay: Double = 1e-6
                    )

case class Ratings(userId: Int, itemId: Int, label: Int)

object NCFExample {

  def main(args: Array[String]): Unit = {

    val defaultParams = NCFParams()

    val parser = new OptionParser[NCFParams]("NCF Example") {
      opt[String]("inputDir")
        .text(s"inputDir")
        .action((x, c) => c.copy(inputDir = x))
      opt[Int]('b', "batchSize")
        .text(s"batchSize")
        .action((x, c) => c.copy(batchSize = x.toInt))
      opt[Int]('e', "nEpochs")
        .text("epoch numbers")
        .action((x, c) => c.copy(nEpochs = x))
      opt[Double]('l', "lRate")
        .text("learning rate")
        .action((x, c) => c.copy(learningRate = x.toDouble))
    }

    parser.parse(args, defaultParams).map {
      params =>
        run(params)
    } getOrElse {
      System.exit(1)
    }
  }

  def run(param: NCFParams): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = new SparkConf()
    conf.setAppName("NCFExample").set("spark.sql.crossJoin.enabled", "true")
    val sc = NNContext.getNNContext(conf)
    val sqlContext = SQLContext.getOrCreate(sc)

    val (data, userCount, itemCount) = loadPublicData(sqlContext, param.inputDir)

    val isImplicit = true
    val ncf = NeuralCF[Float](
      userCount = userCount,
      itemCount = itemCount,
      numClasses = 2,
      userEmbed = 20,
      itemEmbed = 20,
      hiddenLayers = Array(40, 20, 10))

    val pairFeatureRdds: RDD[UserItemFeature[Float]] =
      assemblyFeature(isImplicit, data, userCount, itemCount)

    val Array(trainpairFeatureRdds, validationpairFeatureRdds) =
      pairFeatureRdds.randomSplit(Array(0.8, 0.2))
    val trainRdds = trainpairFeatureRdds.map(x => x.sample)
    val validationRdds = validationpairFeatureRdds.map(x => x.sample)

    val optimizer = Optimizer(
      model = ncf,
      sampleRDD = trainRdds,
      criterion = ClassNLLCriterion[Float](),
      batchSize = 1000)

    val optimMethod = new Adam[Float](
      learningRate = 1e-2,
      learningRateDecay = 1e-4)

    optimizer
      .setOptimMethod(optimMethod)
      .setEndWhen(Trigger.maxEpoch(3))
      .optimize()

    val results = ncf.predict(validationRdds)
    results.take(5).foreach(println)
    val resultsClass = ncf.predictClass(validationRdds)
    resultsClass.take(5).foreach(println)

    val userItemPairPrediction = ncf.predictUserItemPair(validationpairFeatureRdds)

    userItemPairPrediction.take(5).foreach(println)

    val userRecs = ncf.recommendForUser(validationpairFeatureRdds, 3)
    val itemRecs = ncf.recommendForItem(validationpairFeatureRdds, 3)

    userRecs.take(10).foreach(println)
    itemRecs.take(10).foreach(println)

    val validationDF = sqlContext.createDataFrame(validationpairFeatureRdds
      .map(x => Ratings(x.userId, x.itemId, x.sample.getData().last.toInt)))
      .toDF("userId", "itemId", "label")

    val resultsDF = sqlContext.createDataFrame(userItemPairPrediction).toDF()

    val evaluationDF = resultsDF.join(validationDF, Array("userId", "itemId"))
      .select(col("userId"), col("itemId"),
        col("label").cast(DoubleType), col("prediction").cast(DoubleType))

    evaluationDF.show(10)
    evaluationDF.groupBy("prediction").count().show()

    val evaluation = new MulticlassClassificationEvaluator().setPredictionCol("prediction")
      .setMetricName("accuracy").evaluate(evaluationDF)
    println("evaluation result on validationDF: " + evaluation)

  }

  def loadPublicData(sqlContext: SQLContext, dataPath: String): (DataFrame, Int, Int) = {
    import sqlContext.implicits._

    val indexedDF = sqlContext.read.text(dataPath + "/ratings.dat").as[String]
      .map(x => {
        val data: Array[Int] = x.split("::").map(n => n.toInt)
        Ratings(data(0), data(1), data(2))
      })
      .toDF()
    //  .filter("userId <=1000 AND itemId <=1000")

    val minMaxRow = indexedDF.agg(min("userId"), max("userId"), min("itemId"), max("itemId"))
      .collect()(0)

    val userCount = minMaxRow.getInt(1)
    val itemCount = minMaxRow.getInt(3)
    indexedDF.show(5)
    (indexedDF, userCount.toInt, itemCount.toInt)
  }

  def assemblyFeature(isImplicit: Boolean = false,
                      indexed: DataFrame,
                      userCount: Int,
                      itemCount: Int): RDD[UserItemFeature[Float]] = {

    val unioned = if (isImplicit) {
      val negativeDF = Utils.getNegativeSamples(indexed, userCount.toInt, itemCount.toInt)
      negativeDF.groupBy("label").count().show()
      negativeDF.unionAll(indexed.withColumn("label", lit(2)))
    }
    else
      indexed

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
