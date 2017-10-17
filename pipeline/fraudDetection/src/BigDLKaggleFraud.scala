
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.ensemble.Bagging
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature.{FuncTransformer, StandardScaler, VectorAssembler}
import org.apache.spark.ml.{DLClassifier, Pipeline}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}

object BigDLKaggleFraud {

  def main(args: Array[String]): Unit = {

    val conf = Engine.createSparkConf()
    val spark = SparkSession.builder().master("local[1]").appName("BigDL Fraud Detection Example:").config(conf).getOrCreate()
    Engine.init

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
    val dlClassifier = new DLClassifier(bigDLModel, criterion, Array(29)).setLabelCol("Class")
      .setBatchSize(10000)
      .setMaxEpoch(200)

    val estimator = new Bagging()
      .setPredictor(dlClassifier)
      .setLabelCol("Class")
      .setIsClassifier(true)
      .setNumModels(20)

    val model = estimator.fit(trainingData)

    val labelConverter2 = new FuncTransformer(udf {d: Double => if (d==2) 0 else d }).setInputCol("Class").setOutputCol("Class")

    (20 to 40).foreach { t =>
      println("thresold: " + t)
      val prediction = model.setThreshold(t).transform(validData)
      val finalData = labelConverter2.transform(prediction).cache()
      evaluate(finalData)
      finalData.unpersist()
    }
  }

  def evaluate(prediction: DataFrame): Unit = {
    val metrics = new BinaryClassificationEvaluator().setRawPredictionCol("prediction").setLabelCol("Class")
    val auPRC = metrics.evaluate(prediction)
    //      println("layers: " + layers.mkString(", "))
    println("Area under precision-recall curve = " + auPRC)
    val recall = new MulticlassClassificationEvaluator()
      .setLabelCol("Class")
      .setMetricName("weightedRecall")
      .evaluate(prediction)
    println("recall = " + recall)

    val precisoin = new MulticlassClassificationEvaluator()
      .setLabelCol("Class")
      .setMetricName("weightedPrecision")
      .evaluate(prediction)
    println("Precision = " + precisoin)

    println("total fraud: " + prediction.filter("Class=1").count())
    println("tp: " + prediction.filter("prediction=1 and Class=1").count())
    println("fp: " + prediction.filter("prediction=1 and Class=0").count())

    println()
    prediction.unpersist()
  }

}
