package com.intel.analytics.zoo.apps.streaming

import com.intel.analytics.bigdl.tensor.Tensor

import scala.collection.immutable._
import scala.io.Source
import org.apache.spark.sql.functions._
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.feature.text.{TextFeature, TextSet}
import com.intel.analytics.zoo.models.textclassification.TextClassifier
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import org.apache.spark.sql.SparkSession
import scopt.OptionParser



object StructuredStreamingTextClassification {

  def main(args: Array[String]) {
    val parser = new OptionParser[TextClassificationParams]("TextClassification Example") {
      opt[String]('h', "host")
        .text("host for network connection")
        .action((x, c) => c.copy(host = x))
      opt[Int]('p', "port")
        .text("Port for network connection")
        .action((x, c) => c.copy(port = x))
      opt[String]("indexPath")
        .text("Path of word to index text file")
        .action((x, c) => c.copy(indexPath = x))
      opt[String]("embeddingPath")
        .required()
        .text("The directory for GloVe embeddings")
        .action((x, c) => c.copy(embeddingPath = x))
      opt[Int]("classNum")
        .text("The number of classes to do classification")
        .action((x, c) => c.copy(classNum = x))
      opt[Int]("partitionNum")
        .text("The number of partitions to cut the dataset into")
        .action((x, c) => c.copy(partitionNum = x))
      opt[Int]("tokenLength")
        .text("The size of each word vector, 50 or 100 or 200 or 300 for GloVe")
        .action((x, c) => c.copy(tokenLength = x))
      opt[Int]("sequenceLength")
        .text("The length of each sequence")
        .action((x, c) => c.copy(sequenceLength = x))
      opt[Int]("maxWordsNum")
        .text("The maximum number of words to be taken into consideration")
        .action((x, c) => c.copy(maxWordsNum = x))
      opt[String]("encoder")
        .text("The encoder for the input sequence, cnn or lstm or gru")
        .action((x, c) => c.copy(encoder = x))
      opt[Int]("encoderOutputDim")
        .text("The output dimension of the encoder")
        .action((x, c) => c.copy(encoderOutputDim = x))
      opt[Int]('b', "batchSize")
        .text("The number of samples per gradient update")
        .action((x, c) => c.copy(batchSize = x))
      opt[String]('m', "model")
        .text("Model snapshot location if any")
        .action((x, c) => c.copy(model = Some(x)))
    }

    parser.parse(args, TextClassificationParams()).map { param =>
      val sc = NNContext.initNNContext("Structured Network Text Streaming Predict")
      val spark = SparkSession
        .builder.config(sc.getConf)
        .getOrCreate()

      import spark.implicits._

      val wordIndex: Map[String, Int] = readWordIndex(param.indexPath)
      val model = TextClassifier.loadModel[Float](param.model.get)
      
      val lines = spark.readStream
        .format("socket")
        .option("host", param.host)
        .option("port", param.port)
        .load()

      // Predict UDF
      val predictTextUDF = udf ( (x: String) => {
        val dataSet = TextSet.array(Array(TextFeature.apply(x)))
        // Pre-processing
        val transformed = dataSet.setWordIndex(wordIndex)
          .tokenize()
          .normalize()
          .word2idx(removeTopN = 10, maxWordsNum = param.maxWordsNum)
          .shapeSequence(param.sequenceLength).generateSample()
        // Predict
        val predictSet = model.predict(transformed,
          batchPerThread = param.partitionNum)
        // Print result
        predictSet.toLocal().array.map(_.getPredict)
          .head.asInstanceOf[Tensor[Float]].toArray()
        // Top 5 is not suit for console print
//        predictSet.toLocal().array.map(_.getPredict)
//          .take(5).asInstanceOf[Array[Tensor[Float]]].map(_.toArray()).toSeq
      })

      val predicts = lines.as[String].withColumn("Predict",
        predictTextUDF(col("value")))

      val query = predicts.writeStream
        .outputMode("append")
        .format("console")
        .start()

      query.awaitTermination()
    }
  }


  def word2index(word: String, wordIndex: Map[String, Int]): Int = {
    wordIndex.apply(word)
  }

  def readWordIndex(path: String): Map[String, Int] = {
    Source.fromFile(path).getLines.map { x =>
      val token = x.split(" ")
      (token(0), token(1).toInt)
    }.toMap
  }
}
