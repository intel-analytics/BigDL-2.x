package com.intel.analytics.zoo.apps.model.inference.spark

import java.util

import com.intel.analytics.zoo.pipeline.inference.{InferenceSupportive, JTensor}
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.JavaConverters._
import java.util.{List => JList}

import scala.collection.mutable.ListBuffer


object SparkBatchExample extends InferenceSupportive {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local[*]").setAppName("text-classification")
    val sc = new SparkContext(conf)

    LogManager.getRootLogger().setLevel(Level.INFO)

    val supportedConcurrentNum = 4
    val stopWordsCount = 10
    val sequenceLength = 200
    val embeddingFilePath = "/home/glorysdj/DEV/data/glove/glove.6B.100d.txt"
    val modelPath = "/home/glorysdj/DEV/models/text-classification.bigdl"


    val model = new TextClassificationInferenceModel(supportedConcurrentNum, stopWordsCount, sequenceLength, embeddingFilePath)
    model.doLoad(modelPath)
    println(model)

    val inputTensor: JTensor = model.preprocess("hello world")
    val input: JList[JTensor] = List(inputTensor).asJava
    val inputs = new util.ArrayList[JList[JTensor]]()
    inputs.add(input)
    println("################" + model.doPredict(inputs))

    val modelVar = sc.broadcast(model)

    val numPartitions = 64

    val count = timing("############## map each") {
      sc.parallelize(List.range(1, 2000)).repartition(numPartitions).map(record => {
        val model = modelVar.value
        val inputTensor: JTensor = model.preprocess(s"hello world, $record")
        val input: JList[JTensor] = List(inputTensor).asJava
        val inputs = new util.ArrayList[JList[JTensor]]()
        inputs.add(input)
        val result = model.doPredict(inputs)
        //println("*******************", record, result)
        result
      }).count
    }
    println(s"count: $count")

    val count2 = timing("############### map partition") {
      sc.parallelize(List.range(1, 2000)).repartition(numPartitions).mapPartitions(partition => {
        val model = modelVar.value
        var records = ListBuffer[JList[JList[JTensor]]]()
        val inputs = new util.ArrayList[JList[JTensor]]()
        while (partition.hasNext) {
          val record = partition.next
          val inputTensor: JTensor = model.preprocess(s"hello world, $record")
          val input: JList[JTensor] = List(inputTensor).asJava
          inputs.add(input)
        }
        val result = model.doPredict(inputs)
        //println("*******************", result)
        records += result
        records.iterator;
      }).count
    }
    println(s"count2: $count2")
  }

}
