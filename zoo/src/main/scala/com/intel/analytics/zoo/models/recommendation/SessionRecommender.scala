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

import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.Sum
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.models.common.ZooModel
import com.intel.analytics.zoo.pipeline.api.keras.layers._
import com.intel.analytics.zoo.pipeline.api.keras.models.Model
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

/**
 * The session based recommender model.
 *
 * @param itemCount       The number of distinct items. Positive integer.
 * @param itemEmbed       The output size of embedding layer. Positive integer.
 * @param mlpHiddenLayers Units of hidden layers for the mlp model. Array of positive integers. Default is Array(20, 20, 20).
 * @param seqLength       The max number of items in the sequence of a session
 * @param rnnHiddenLayers Units of hidden layers for the mlp model. Array of positive integers. Default is Array(20, 20).
 * @param includeHistory  Whether to include purchase history. Boolean. Default is true.
 * @param hisLength       The max number of items in the sequence of historical purchase
 */
class SessionRecommender[T: ClassTag](
                                       val itemCount: Int,
                                       val itemEmbed: Int = 300,
                                       val rnnHiddenLayers: Array[Int] = Array(20, 20),
                                       val seqLength: Int = 5,
                                       val includeHistory: Boolean = true,
                                       val mlpHiddenLayers: Array[Int] = Array(20, 20, 20),
                                       val hisLength: Int = 10
                                     )(implicit ev: TensorNumeric[T]) extends Recommender[T] {

  override def buildModel(): AbstractModule[Tensor[T], Tensor[T], T] = {
    val inputRnn: ModuleNode[T] = Input(inputShape = Shape(seqLength))

    // item embedding layer
    val sessionTable = Embedding[T](itemCount + 1, itemEmbed, "normal", inputLength = seqLength).inputs(inputRnn)

    // rnn part
    var gru = GRU[T](rnnHiddenLayers(0), returnSequences = true).inputs(sessionTable)
    for (i <- 1 until rnnHiddenLayers.length - 1) {
      gru = GRU[T](rnnHiddenLayers(i), returnSequences = true).inputs(gru)
    }
    val gruLast = GRU[T](rnnHiddenLayers.last, returnSequences = false).inputs(gru)
    val rnn = Dense[T](itemCount).inputs(gruLast)

    // mlp part
    if (includeHistory) {
      val inputMlp: ModuleNode[T] = Input(inputShape = Shape(hisLength))
      val hisTable = Embedding[T](itemCount + 1, itemEmbed, inputLength = hisLength).inputs(inputMlp)
      val sum = new KerasLayerWrapper[T](Sum[T](2)).inputs(hisTable)
      var mlp = Dense(mlpHiddenLayers(0), activation = "relu").inputs(sum)
      for (i <- 1 until mlpHiddenLayers.length) {
        mlp = Dense(mlpHiddenLayers(i), activation = "relu").inputs(mlp)
      }
      val mlpLast = Dense(itemCount).inputs(mlp)
      // combine rnn and mlp
      val merged = Merge.merge[T](List(mlpLast, rnn), "sum")
      val out = Activation("softmax").inputs(merged)
      Model(Array(inputMlp, inputRnn), out).asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
    }
    else {
      val out = Activation("softmax").inputs(rnn)
      Model(inputRnn, out).asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
    }
  }

  override def recommendForUser(featureRdd: RDD[UserItemFeature[T]],
                                maxItems: Int): RDD[UserItemPrediction] = {
    throw new UnsupportedOperationException(s"user item: Unimplemented method")
  }

}

object SessionRecommender {
  /**
   * The factory method to create a SessionRecommender instance.
   *
   * @param itemCount       The number of distinct items. Positive integer.
   * @param itemEmbed       The output size of embedding layer. Positive integer.
   * @param mlpHiddenLayers Units of hidden layers for the mlp model. Array of positive integers. Default is Array(20, 20, 20).
   * @param seqLength       The max number of items in the sequence of a session
   * @param rnnHiddenLayers Units of hidden layers for the mlp model. Array of positive integers. Default is Array(20, 20).
   * @param includeHistory  Whether to include purchase history. Boolean. Default is true.
   * @param hisLength       The max number of items in the sequence of historical purchase
   */
  def apply[@specialized(Float, Double) T: ClassTag](
                                                      itemCount: Int,
                                                      itemEmbed: Int = 100,
                                                      rnnHiddenLayers: Array[Int] = Array(20, 20),
                                                      seqLength: Int = 5,
                                                      includeHistory: Boolean = true,
                                                      mlpHiddenLayers: Array[Int] = Array(20, 20, 20),
                                                      hisLength: Int = 10)
                                                    (implicit ev: TensorNumeric[T]): SessionRecommender[T] = {
    new SessionRecommender[T](
      itemCount,
      itemEmbed,
      rnnHiddenLayers,
      seqLength,
      includeHistory,
      mlpHiddenLayers,
      hisLength
    ).build()
  }

  /**
   * Load an existing SessionRecommender model (with weights).
   *
   * @param path       The path for the pre-defined model.
   *                   Local file system, HDFS and Amazon S3 are supported.
   *                   HDFS path should be like "hdfs://[host]:[port]/xxx".
   *                   Amazon S3 path should be like "s3a://bucket/xxx".
   * @param weightPath The path for pre-trained weights if any. Default is null.
   * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
   */
  def loadModel[T: ClassTag](
                              path: String,
                              weightPath: String = null
                            )(implicit ev: TensorNumeric[T]): SessionRecommender[T] = {
    ZooModel.loadModel(path, weightPath).asInstanceOf[SessionRecommender[T]]
  }
}
