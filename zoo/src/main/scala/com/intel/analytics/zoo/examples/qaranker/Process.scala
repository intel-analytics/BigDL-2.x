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

package com.intel.analytics.zoo.examples.qaranker

import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.feature.common.Relation
import org.apache.spark.sql.SQLContext

object Process {
  def main(args: Array[String]): Unit = {
    val sc = NNContext.initNNContext("Transform raw data")
    val targetRelations = Array(Relation("Q1", "A1", 1), Relation("Q1", "A2", 0),
      Relation("Q2", "A1", 0), Relation("Q2", "A2", 1))
    val relations = sc.parallelize(targetRelations)
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._
//    val df = relations.toDF()
//    df.write.parquet("/home/kai/relations.parquet")
//    val df = sqlContext.read.parquet("/home/kai/question_corpus.parquet")
//    print("1")

//    val questions = Array(Text("Q1", "what is your project?"), Text("Q2", "how old are you?"))
//    val df2 = sc.parallelize(questions).toDF()
//    df2.write.parquet("/home/kai/question_corpus.parquet")
    val questions = Array(Text("A1", "Analytics Zoo."),
  Text("A2", "I am 18 years old."))
//    println(questions(0).text)
    val df2 = sc.parallelize(questions).toDF()
    df2.write.parquet("/home/kai/answer_corpus.parquet")
  }
}

case class Text(id: String, text: String)
