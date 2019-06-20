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
import com.intel.analytics.zoo.models.common.{KerasZooModel, ZooModel}
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

case class UserItemFeature[T: ClassTag](userId: Int, itemId: Int, sample: Sample[T])

case class UserItemPrediction(userId: Int, itemId: Int, prediction: Int, probability: Double)

/**
 * The base class for recommendation models in Analytics Zoo.
 *
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
abstract class Recommender[T: ClassTag](implicit ev: TensorNumeric[T])
  extends KerasZooModel[Tensor[T], Tensor[T], T] {

  /**
   * Recommend a number of items for each user given a rdd of user item pair features.
   *
   * @param featureRdd RDD of user item pair feature.
   * @param maxItems   Number of items to be recommended to each user. Positive integer.
   * @return RDD of user item pair prediction.
   */
  def recommendForUser(featureRdd: RDD[UserItemFeature[T]],
                       maxItems: Int): RDD[UserItemPrediction] = {

    val pairPredictions = predictUserItemPair(featureRdd)

    pairPredictions
      .map(x => (x.userId, x))
      .groupByKey()
      .flatMap(x => {
        val ordered = x._2.toArray.sortBy(y => (-y.prediction, -y.probability)).take(maxItems)
        ordered
      })
  }

  /**
   * Recommend a number of users for each item given a rdd of user item pair features.
   *
   * @param featureRdd RDD of user item pair feature.
   * @param maxUsers   Number of users to be recommended to each item. Positive integer.
   * @return RDD of user item pair prediction.
   */
  def recommendForItem(featureRdd: RDD[UserItemFeature[T]],
                       maxUsers: Int): RDD[UserItemPrediction] = {
    val pairPredictions = predictUserItemPair(featureRdd)

    pairPredictions
      .map(x => (x.itemId, x))
      .groupByKey()
      .flatMap(x => {
        val ordered = x._2.toArray.sortBy(y => (-y.prediction, -y.probability)).take(maxUsers)
        ordered
      })
  }

  /**
   * Predict for user-item pairs.
   *
   * @param featureRdd RDD of user item pair feature.
   * @return RDD of user item pair prediction.
   */
  def predictUserItemPair(featureRdd: RDD[UserItemFeature[T]]): RDD[UserItemPrediction] = {
    featureRdd.persist()
    val inputCount = featureRdd.count()
    val idPairs = featureRdd.map(pair => (pair.userId, pair.itemId))
    val features = featureRdd.map(pair => pair.sample)
    val raw = predict(features)
    val predictProb = raw.map { x =>
      val _output = x.toTensor[T]
      val predict: Int = ev.toType[Int](_output.max(1)._2.valueAt(1))
      val probability = _output.valueAt(predict).asInstanceOf[Float]
      (predict, probability)
    }
    val outRdd: RDD[UserItemPrediction] = idPairs.zip(predictProb)
      .map(x => UserItemPrediction(x._1._1, x._1._2, x._2._1, x._2._2)).cache()
    featureRdd.unpersist()

    require(inputCount == outRdd.count(), s"count of features must equal to count of prediction")
    outRdd
  }
}
