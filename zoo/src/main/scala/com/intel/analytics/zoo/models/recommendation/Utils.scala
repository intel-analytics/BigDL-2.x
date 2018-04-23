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

package com.intel.analytics.zoo.models.recommendation

import org.apache.spark.sql.DataFrame

import scala.util.Random

object Utils {

  def getNegativeSamples(indexed: DataFrame, userCount: Int, itemCount: Int): DataFrame = {
    val schema = indexed.schema
    require(schema.fieldNames.contains("userId"), s"Column userId should exist")
    require(schema.fieldNames.contains("itemId"), s"Column itemId should exist")
    require(schema.fieldNames.contains("label"), s"Column label should exist")

    val indexedDF = indexed.select("userId", "itemId", "label")

    val sampleDict = indexedDF.rdd.map(row => row(0) + "," + row(1)).collect().toSet

    val dfCount = indexedDF.count.toInt

    import indexed.sqlContext.implicits._

    val ran = new Random(seed = 42L)
    val negative = indexedDF.rdd.map(x => {
      val uid = x.getAs[Int](0)
      val iid = Math.max(ran.nextInt(itemCount), 1)
      (uid, iid)
    })
      .filter(x => !sampleDict.contains(x._1 + "," + x._2)).distinct()
      .map(x => (x._1, x._2, 1))
      .toDF("userId", "itemId", "label")

    negative
  }
}
