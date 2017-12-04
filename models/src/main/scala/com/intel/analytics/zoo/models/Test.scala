/*
 * Copyright 2016 The BigDL Authors.
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

package com.intel.analytics.zoo.models

import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.zoo.models.alexnet.AlexnetPredictor
import com.intel.analytics.zoo.models.dataset.PredictResult
import com.intel.analytics.zoo.models.inception.InceptionV1Predictor
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{SQLContext, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.JavaConverters._

object Test {


  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local[1]").setAppName("Test distributed predict")
      .set("spark.shuffle.reduceLocality.enabled", "false")
      .set("spark.speculation", "false")
      .set("spark.shuffle.blockTransferService", "nio")
      .set("spark.scheduler.minRegisteredResourcesRatio", "1.0")
    val sc = new SparkContext(conf)
    val sqlContext = SQLContext.getOrCreate(sc)
    Engine.init

    val predictor = InceptionV1Predictor("/home/jerry/data/model_zoo/inceptionv1/inceptionv1.bigdl")
    // val res = predictor.
    //  predictLocal("/home/jerry/data/test_img/val/first/ILSVRC2012_val_00048969.JPEG", 2)
   val paths = sc.parallelize(Seq("/home/jerry/Downloads/cat.jpeg",
   "/home/jerry/Downloads/fish.jpg"))

    val res = predictor.predictDistributed(paths, 3)

    val c = res.collect()

    val other: RDD[Float] = null

    val df = sqlContext.createDataFrame(res).toDF("id", "clsWithCredit")
    df.printSchema()
    df.show()
    println()
  }

/*
  def main(args: Array[String]): Unit = {
    /*
    System.getProperties.asScala.foreach(p
      => println(p._1 + "=" + p._2))
    val predictor = AlexnetPredictor("/home/jerry/data/model_zoo/alexnet/alexnet.bigdl",
      "/home/jerry/Downloads/mean.txt")
    val res = predictor.predictLocal("/home/jerry/Downloads/cat.jpeg", 2)
    println()
    */
    val data = Seq("x", "y", "z")
    data.zipWithIndex.foreach(d => {
      println(data(d._2))
    })
  }
  */
}
