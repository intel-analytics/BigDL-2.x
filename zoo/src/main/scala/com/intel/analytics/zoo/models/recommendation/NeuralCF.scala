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

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.models.common.ZooModel

import scala.reflect.ClassTag

/**
 * The neural collaborative filtering model used for recommendation.
 *
 * @param userCount    The number of users. Positive integer.
 * @param itemCount    The number of items. Positive integer.
 * @param numClasses   The number of classes. Positive integer.
 * @param userEmbed    Units of user embedding. Positive integer. Default is 20.
 * @param itemEmbed    Units of item embedding. Positive integer. Default is 20.
 * @param hiddenLayers Units hiddenLayers for MLP. Array of positive integers.
 *                     Default is Array(40, 20, 10).
 * @param includeMF    Whether to include Matrix Factorization. Boolean. Default is true.
 * @param mfEmbed      Units of matrix factorization embedding. Positive integer.
 *                     Default is 20.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */

class NeuralCF[T: ClassTag](
    val userCount: Int,
    val itemCount: Int,
    val numClasses: Int,
    val userEmbed: Int = 20,
    val itemEmbed: Int = 20,
    val hiddenLayers: Array[Int] = Array(40, 20, 10),
    val includeMF: Boolean = true,
    val mfEmbed: Int = 20)(implicit ev: TensorNumeric[T])
  extends Recommender[T] {

  override def buildModel(): AbstractModule[Tensor[T], Tensor[T], T] = {
    val model = Sequential[T]()

    val mlpUserTable = LookupTable[T](userCount, userEmbed)
    val mlpItemTable = LookupTable[T](itemCount, itemEmbed)
    mlpUserTable.setWeightsBias(Array(Tensor[T](userCount, userEmbed).randn(0, 0.1)))
    mlpItemTable.setWeightsBias(Array(Tensor[T](itemCount, itemEmbed).randn(0, 0.1)))
    val mlpEmbeddedLayer = Concat[T](2)
      .add(Sequential[T]().add(Select(2, 1)).add(mlpUserTable))
      .add(Sequential[T]().add(Select(2, 2)).add(mlpItemTable))
    val mlpModel = Sequential[T]()
    mlpModel.add(mlpEmbeddedLayer)
    val linear1 = Linear[T](itemEmbed + userEmbed, hiddenLayers(0))
    mlpModel.add(linear1).add(ReLU())
    for (i <- 1 to hiddenLayers.length - 1) {
      mlpModel.add(Linear(hiddenLayers(i - 1), hiddenLayers(i))).add(ReLU())
    }

    if (includeMF) {
      require(mfEmbed > 0, s"please provide meaningful number of embedding units")
      val mfUserTable: LookupTable[T] = LookupTable[T](userCount, mfEmbed)
      val mfItemTable = LookupTable[T](itemCount, mfEmbed)
      mfUserTable.setWeightsBias(Array(Tensor[T](userCount, mfEmbed).randn(0, 0.1)))
      mfItemTable.setWeightsBias(Array(Tensor[T](itemCount, mfEmbed).randn(0, 0.1)))
      val mfEmbeddedLayer = ConcatTable()
        .add(Sequential[T]().add(Select(2, 1)).add(mfUserTable))
        .add(Sequential[T]().add(Select(2, 2)).add(mfItemTable))
      val mfModel = Sequential[T]()
      mfModel.add(mfEmbeddedLayer).add(CMulTable())
      val concatedModel = Concat(2).add(mfModel).add(mlpModel)
      model.add(concatedModel)
        .add(Linear(mfEmbed + hiddenLayers.last, numClasses))
    }
    else {
      model.add(mlpModel).
        add(Linear(hiddenLayers.last, numClasses))
    }
    model.add(LogSoftMax[T]())

    model.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object NeuralCF {
  /**
   * The factory method to create a NeuralCF instance.
   */
  def apply[@specialized(Float, Double) T: ClassTag](
      userCount: Int,
      itemCount: Int,
      numClasses: Int,
      userEmbed: Int = 20,
      itemEmbed: Int = 20,
      hiddenLayers: Array[Int] = Array(40, 20, 10),
      includeMF: Boolean = true,
      mfEmbed: Int = 20)(implicit ev: TensorNumeric[T]): NeuralCF[T] = {
    new NeuralCF[T](userCount, itemCount, numClasses, userEmbed,
      itemEmbed, hiddenLayers, includeMF, mfEmbed).build()
  }

  /**
   * Load an existing NeuralCF model (with weights).
   *
   * @param path The path for the pre-defined model.
   *             Local file system, HDFS and Amazon S3 are supported.
   *             HDFS path should be like "hdfs://[host]:[port]/xxx".
   *             Amazon S3 path should be like "s3a://bucket/xxx".
   * @param weightPath The path for pre-trained weights if any. Default is null.
   * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
   */
  def loadModel[T: ClassTag](
      path: String,
      weightPath: String = null)(implicit ev: TensorNumeric[T]): NeuralCF[T] = {
    ZooModel.loadModel(path, weightPath).asInstanceOf[NeuralCF[T]]
  }
}

