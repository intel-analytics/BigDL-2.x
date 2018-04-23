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

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.models.common.ZooModel
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{col, desc, rank}

import scala.reflect.ClassTag

case class UserItemFeature[T: ClassTag](userId: Int, itemId: Int, sample: Sample[T])

case class UserItemPrediction(userId: Int, itemId: Int, prediction: Int, probability: Double)

/**
 * The factory for recommender.
 *
 * @param userCount The number of users. Positive integer.
 * @param itemCount The number of items. Positive integer.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
abstract class Recommender[T: ClassTag](userCount: Int, itemCount: Int
                                       )(implicit ev: TensorNumeric[T])
  extends ZooModel[Tensor[T], Tensor[T], T] {

  /**
   * Recommend a number of items for each user given a rdd of user item pair features
   *
   * @param featureRdd RDD of user item pair feature.
   * @param maxItems Number of items to be recommended to each user. Positive integer.
   * @return RDD of user item pair prediction.
   */
  def recommendForUser(featureRdd: RDD[UserItemFeature[T]],
                       maxItems: Int): RDD[UserItemPrediction] = {

    val results = predictUserItemPair(featureRdd)
    val sqlContext = SQLContext.getOrCreate(results.sparkContext)
    val resultsDf = sqlContext.createDataFrame(results).toDF()

    val window = Window.partitionBy("userId").orderBy(desc("prediction"), desc("probability"))
    val out = resultsDf.withColumn("rank", rank.over(window))
      .where(col("rank") <= maxItems)
      .drop("rank")
      .rdd.map(row => UserItemPrediction
    (row.getAs[Int](0), row.getAs[Int](1), row.getAs[Int](2), row.getAs[Double](3)))
    out
  }

  /**
   * Recommend a number of users for each item given a rdd of user item pair features
   *
   * @param featureRdd RDD of user item pair feature.
   * @param maxUsers Number of users to be recommended to each item. Positive integer.
   * @return RDD of user item pair prediction.
   */
  def recommendForItem(featureRdd: RDD[UserItemFeature[T]],
                       maxUsers: Int): RDD[UserItemPrediction] = {
    val results = predictUserItemPair(featureRdd)
    val sqlContext = SQLContext.getOrCreate(results.sparkContext)
    val resultsDf = sqlContext.createDataFrame(results).toDF()

    val window = Window.partitionBy("itemId").orderBy(desc("prediction"), desc("probability"))
    val out = resultsDf.withColumn("rank", rank.over(window))
      .where(col("rank") <= maxUsers)
      .drop("rank")
      .rdd.map(row => UserItemPrediction(
      row.getAs[Int](0), row.getAs[Int](1), row.getAs[Int](2), row.getAs[Double](3)))
    out
  }

  /**
   * Predict for and user item pair
   *
   * @param featureRdd RDD of user item pair feature.
   * @return RDD of user item pair prediction.
   */
  def predictUserItemPair(featureRdd: RDD[UserItemFeature[T]]): RDD[UserItemPrediction]

}
