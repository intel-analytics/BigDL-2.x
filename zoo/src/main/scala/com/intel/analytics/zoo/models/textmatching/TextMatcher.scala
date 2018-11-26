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

package com.intel.analytics.zoo.models.textmatching

import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.feature.text.{DistributedTextSet, LocalTextSet, TextSet}
import com.intel.analytics.zoo.models.common.ZooModel
import org.apache.log4j.Logger

import scala.reflect.ClassTag
import scala.util.Random

/**
 * The base class for text matching models in Analytics Zoo.
 * Referred to MatchZoo implementation: https://github.com/NTMC-Community/MatchZoo
 */
abstract class TextMatcher[T: ClassTag](
    val text1Length: Int,
    val vocabSize: Int,
    val embedSize: Int = 300,
    val embedWeights: Tensor[T] = null,
    val trainEmbed: Boolean = true)(implicit ev: TensorNumeric[T])
  extends ZooModel[Activity, Activity, T] {

  import TextMatcher.logger

  def evaluate(
      textSet: TextSet,
      metrics: (Tensor[T], Tensor[T]) => Double): Double = {
    val result = textSet match {
      case distributed: DistributedTextSet =>
        val rdd = distributed.rdd
        val modelBroad = ModelBroadcast[T]().broadcast(rdd.sparkContext, setEvaluateStatus())
        rdd.mapPartitions(partition => {
          val localModel = modelBroad.value()
          partition.map(feature => {
            val input = feature.getSample.feature()
            val output = localModel.forward(input).toTensor[T]
            val target = feature.getSample.label().toTensor[T]
            metrics(output, target)
          })
        }).mean()
      case local: LocalTextSet =>
        val res = local.array.map(feature => {
          val input = feature.getSample.feature()
          val output = setEvaluateStatus().forward(input).toTensor[T]
          val target = feature.getSample.label().toTensor[T]
          metrics(output, target)
        })
        res.sum / res.length
    }
    result
  }

  def evaluateMAP(
      textSet: TextSet,
      threshold: Double = 0.0): Double = {
    val map = evaluate(textSet, TextMatcher.map(threshold))
    logger.info(s"map: $map")
    map
  }

  def evaluateNDCG(
      textSet: TextSet,
      k: Int,
      threshold: Double = 0.0): Double = {
    val ndcg = evaluate(textSet, TextMatcher.ndcg(k, threshold))
    logger.info(s"ndcg@$k: $ndcg")
    ndcg
  }
}

object TextMatcher {

  val logger: Logger = Logger.getLogger(getClass)

  def ndcg[@specialized(Float, Double) T: ClassTag](
      k: Int, threshold: Double = 0.0)
    (implicit ev: TensorNumeric[T]): (Tensor[T], Tensor[T]) => Double = {
    require(k > 0, s"k for NDCG should be a positive integer, but got $k")
    def validate(output: Tensor[T], target: Tensor[T])
      (implicit ev: TensorNumeric[T]): Double = {
      require(output.size().length == 2 && output.size()(1) == 1,
        s"output should be of shape (batch, 1), but got ${output.size()}")
      require(target.size().length == 2 && target.size()(1) == 1,
        s"target should be of shape (batch, 1), but got ${target.size()}")
      val yTrue = target.squeezeNewTensor().toArray().map(ev.toType[Double])
      val yPred = output.squeezeNewTensor().toArray().map(ev.toType[Double])
      val c = Random.shuffle(yTrue.zip(yPred).toList)
      val c_g = c.sortBy(_._1).reverse
      val c_p = c.sortBy(_._2).reverse
      var idcg = 0.0
      var dcg = 0.0
      for (((g, p), i) <- c_g.zipWithIndex) {
        if (i < k && g > threshold) {
          idcg += math.pow(2.0, g) / math.log(2.0 + i)
        }
      }
      for (((g, p), i) <- c_p.zipWithIndex) {
        if (i < k && g > threshold) {
          dcg += math.pow(2.0, g) / math.log(2.0 + i)
        }
      }
      if (idcg == 0.0) 0.0 else dcg / idcg
    }
    validate
  }

  def map[@specialized(Float, Double) T: ClassTag](
      threshold: Double = 0.0)
    (implicit ev: TensorNumeric[T]): (Tensor[T], Tensor[T]) => Double = {
    def validate(output: Tensor[T], target: Tensor[T])
      (implicit ev: TensorNumeric[T]): Double = {
      require(output.size().length == 2 && output.size()(1) == 1,
        s"output should be of shape (batch, 1), but got ${output.size()}")
      require(target.size().length == 2 && target.size()(1) == 1,
        s"target should be of shape (batch, 1), but got ${target.size()}")
      val yTrue = target.squeezeNewTensor().toArray().map(ev.toType[Double])
      val yPred = output.squeezeNewTensor().toArray().map(ev.toType[Double])
      val c = Random.shuffle(yTrue.zip(yPred).toList).sortBy(_._2).reverse
      var s = 0.0
      var ipos = 0
      for (((g, p), i) <- c.zipWithIndex) {
        if (g > threshold) {
          ipos += 1
          s += ipos / (i + 1.0)
        }
      }
      if (ipos == 0) 0.0
      else s / ipos
    }
    validate
  }
}
