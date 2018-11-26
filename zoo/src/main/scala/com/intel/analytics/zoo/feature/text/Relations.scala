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

package com.intel.analytics.zoo.feature.text

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext

import scala.io.Source

object Relations {

  def readParquet(path: String, sqlContext: SQLContext): RDD[Relation] = {
    sqlContext.read.parquet(path).rdd.map(row => {
      val text1ID = row.getAs[String]("text1ID")
      val text2ID = row.getAs[String]("text2ID")
      val label = row.getAs[Int]("label")
      Relation(text1ID, text2ID, label)
    })
  }

  // Without header
  // QID, AID, label
  def readCSV(path: String, sc: SparkContext, minPartitions: Int = 1): RDD[Relation] = {
    sc.textFile(path, minPartitions).map(line => {
      val subs = line.split(",")
      Relation(subs(0), subs(1), subs(2).toInt)
    })
  }

  def readCSV(path: String): Array[Relation] = {
    val src = Source.fromFile(path)
    src.getLines().toArray.map(line => {
      val subs = line.split(",")
      Relation(subs(0), subs(1), subs(2).toInt)
    })
  }

  def generateRelationPairs(relations: RDD[Relation]): RDD[RelationPair] = {
    val positive = relations.filter(_.label == 1).groupBy(_.text1ID)
    val negative = relations.filter(_.label == 0).groupBy(_.text1ID)
    positive.cogroup(negative).flatMap(x => {
      val posIDs = x._2._1.flatten.toArray.map(_.text2ID)
      val negIDs = x._2._2.flatten.toArray.map(_.text2ID)
      posIDs.flatMap(y => negIDs.map(z => RelationPair(x._1, y, z)))
    })
  }
}

case class Relation(text1ID: String, text2ID: String, label: Int)

case class RelationPair(text1ID: String, text2PosID: String, text2NegID: String)
