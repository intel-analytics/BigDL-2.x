package com.intel.analytics.bigdl.zoo.fraud

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericDouble
import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature.{FuncTransformer, StandardScaler, StratifiedSampler, VectorAssembler}
import org.apache.spark.ml.{DLClassifier, DLModel, Pipeline}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, StructField}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

object BigDLKaggleFraud {

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("com").setLevel(Level.WARN)
    val conf = Engine.createSparkConf()
    val spark = SparkSession.builder().master("local[1]").appName("BigDL Fraud Detection Example:").config(conf).getOrCreate()
    Engine.init
    import spark.implicits._

    val raw = spark.read
      .format("csv")
      .option("header", "true") //reading the headers
      .option("mode", "DROPMALFORMED")
      .csv("data/creditcard.csv")
    val df = raw.select(((1 to 28).map(i => "V" + i) ++ Array("Time", "Amount", "Class")).map(s => col(s).cast("Double")): _*)

    println("total: " + df.count())
    df.groupBy("Class").count().show()

    val labelConverter = new FuncTransformer(udf {d: Double => if (d==0) 2 else d }).setInputCol("Class").setOutputCol("Class")
    val assembler = new VectorAssembler().setInputCols((1 to 28).map(i => "V" + i).toArray ++ Array("Amount")).setOutputCol("assembled")
    val scaler = new StandardScaler().setInputCol("assembled").setOutputCol("features")
    val pipeline = new Pipeline().setStages(Array(assembler, scaler, labelConverter))
    val pipelineModel = pipeline.fit(df)
    val data = pipelineModel.transform(df)

    val splitTime = data.stat.approxQuantile("Time", Array(0.7), 0.001).head
    val trainingData = data.filter(s"Time<$splitTime").cache()
    val validData = data.filter(s"Time>=$splitTime").cache()
    println("training count: " + trainingData.count())
    println("validData count: " + validData.count())

    val bigDLModel = Sequential()
      .add(Linear(29, 10))
      .add(Linear(10, 2))
      .add(LogSoftMax())
    val criterion = ClassNLLCriterion()

    val numModel = 10
    val models = new Array[DLModel[Double]](numModel)
    for (i <- 0 until numModel) {
      val sampler = new StratifiedSampler(Map(2 -> 0.05, 1-> 10, 0 -> 1)).setLabel("Class")
      val bootstrapSample = sampler.transform(trainingData)
      val singleModel = new DLClassifier(bigDLModel.cloneModule(), criterion.cloneCriterion(), Array(29))
        .setLabelCol("Class")
        .setBatchSize(10000)
        .setLearningRate(3e-2)
        .setMaxEpoch(100)
        .fit(bootstrapSample)
      models(i) = singleModel
      println("model trained: " + i)
    }

    val predicts = models.map { m =>
      m.transform(validData).select("prediction").as[Double].rdd
    }
    val aggPredict: RDD[Double] = predicts.reduce { (rdd1: RDD[Double], rdd2: RDD[Double]) =>
      val result = rdd1.zip(rdd2).map { case (p1, p2) => p1 + p2 }
      result
    }

    (0 to numModel * 2).foreach { threshold =>
      println("threshold: " + threshold)
      val rows = validData.toDF().rdd.zip(aggPredict).map { case (row, p) =>
        val q = if(p >= threshold) 0.0 else 1.0
        Row.fromSeq(row.toSeq ++ Seq(q))
      }

      val predictDF = validData.sparkSession.createDataFrame(rows, validData.schema.add(StructField("prediction", DoubleType)))
      evaluateModel(predictDF)
    }
  }

  def evaluateModel(predictionDF: DataFrame): Unit = {
    predictionDF.cache()
    // convert the prediction and label column back to {0, 1}
    val labelConverter2 = new FuncTransformer(udf {d: Double => if (d==2) 0 else d }).setInputCol("Class").setOutputCol("Class")
    val labelConverter3 = new FuncTransformer(udf {d: Double => if (d==2) 0 else d }).setInputCol("prediction").setOutputCol("prediction")
    val finalData = labelConverter2.transform(labelConverter3.transform(predictionDF))

    val metrics = new BinaryClassificationEvaluator().setRawPredictionCol("prediction").setLabelCol("Class")
    val auPRC = metrics.evaluate(finalData)
    println("\nArea under precision-recall curve: = " + auPRC)

    val recall = new MulticlassClassificationEvaluator().setLabelCol("Class").setMetricName("weightedRecall").evaluate(finalData)
    println("\nrecall = " + recall)

    val precisoin = new MulticlassClassificationEvaluator().setLabelCol("Class").setMetricName("weightedPrecision").evaluate(finalData)
    println("\nPrecision = " + precisoin)
    predictionDF.unpersist()
  }

}
