package com.intel.analytics.zoo.models.recommendation

import org.apache.spark.sql.DataFrame

import scala.util.Random

object Utils {

  def getNegativeSamples(indexed: DataFrame, userCount: Int, itemCount: Int) = {
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
