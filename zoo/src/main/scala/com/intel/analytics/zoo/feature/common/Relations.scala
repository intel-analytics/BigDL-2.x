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

package com.intel.analytics.zoo.feature.common

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext

import scala.io.Source

object Relations {

  def readParquet(path: String, sqlContext: SQLContext): RDD[Relation] = {
    sqlContext.read.parquet(path).rdd.map(row => {
      val id1 = row.getAs[String]("id1")
      val id2 = row.getAs[String]("id2")
      val label = row.getAs[Int]("label")
      Relation(id1, id2, label)
    })
  }

  // Without header
  // id1, id2, label
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
    val positive = relations.filter(_.label == 1).groupBy(_.id1)
    val negative = relations.filter(_.label == 0).groupBy(_.id1)
    positive.cogroup(negative).flatMap(x => {
      val posIDs = x._2._1.flatten.toArray.map(_.id2)
      val negIDs = x._2._2.flatten.toArray.map(_.id2)
      posIDs.flatMap(y => negIDs.map(z => RelationPair(x._1, y, z)))
    })
  }
}

/**
 * It represents the relationship between two items.
 */
case class Relation(id1: String, id2: String, label: Int)

/**
 * A relation pair is made up of two relations:
 * Relation(id1, id2Positive, label>0)
 * Relation(id1, id2Negative, label=0)
 */
case class RelationPair(id1: String, id2Positive: String, id2Negative: String)
