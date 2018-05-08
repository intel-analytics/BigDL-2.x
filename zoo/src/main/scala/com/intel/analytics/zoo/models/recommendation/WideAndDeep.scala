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

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.models.common.ZooModel

import scala.reflect.ClassTag

/**
 * Wide and deep model and its feature generation part share the same data information
 * @param wideBaseCols   Data of wideBaseCols together with wideCrossCols will be fed into
 *                       wide model.
 * @param wideBaseDims   Dimension of wideBaseCols, dimension of data in wideBaseCols should be
 *                       with in the range of wideBaseDims.
 * @param wideCrossCols  Data of wideCrossCols will be fed into wide model.
 * @param wideCrossDims  Dimension of crossed columns, dimension of data in wideCrossCols should
 *                       be with in the range of wideCrossDims.
 * @param indicatorCols  Data of indicatorCols will be fed into deep model as multi-hot vectors.
 * @param indicatorDims  Dimension indicatorCols, dimension of data in indicatorCols should be
 *                       with in the range of indicatorDims.
 * @param embedCols      Data of embedCols will be fed into deep model as embeddings.
 * @param embedInDims    Input dimension of the data in embedCols,
 *                       dimension of data in embedCols should be within the range of embedInDims.
 * @param embedOutDims   Dimension of embeddings
 * @param continuousCols Data of continuousCols is treated as continuous values for deep model.
 * @param label          Name of label column.
 */
case class ColumnFeatureInfo(wideBaseCols: Array[String] = Array[String](),
                             wideBaseDims: Array[Int] = Array[Int](),
                             wideCrossCols: Array[String] = Array[String](),
                             wideCrossDims: Array[Int] = Array[Int](),
                             indicatorCols: Array[String] = Array[String](),
                             indicatorDims: Array[Int] = Array[Int](),
                             embedCols: Array[String] = Array[String](),
                             embedInDims: Array[Int] = Array[Int](),
                             embedOutDims: Array[Int] = Array[Int](),
                             continuousCols: Array[String] = Array[String](),
                             label: String = "label") extends Serializable

/**
 * The wide and deep model for recommendation.*
 * @param modelType      String, "wide", "deep", "wide_n_deep" are supported.
 * @param numClasses     The number of classes. Positive integer.
 * @param wideBaseDims   Dimension of wideBaseCols, dimension of data in wideBaseCols should be
 *                       with in the range of wideBaseDims.
 * @param wideCrossDims  Dimension of crossed columns, dimension of data in wideCrossCols should
 *                       be with in the range of wideCrossDims.
 * @param indicatorDims  Dimension indicatorCols, dimension of data in indicatorCols should be
 *                       with in the range of indicatorDims.
 * @param embedInDims    Input dimension of the data in embedCols,
 *                       dimension of data in embedCols should be within the range of embedInDims.
 * @param embedOutDims   Dimension of embeddings
 * @param continuousCols Data of continuousCols is treated as continuous values for deep model.
 * @param hiddenLayers   Units hidenLayers of deep model. Array of positive integer.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class WideAndDeep[T: ClassTag] private(val modelType: String = "wide_n_deep",
                                       val numClasses: Int,
                                       val wideBaseDims: Array[Int] = Array[Int](),
                                       val wideCrossDims: Array[Int] = Array[Int](),
                                       val indicatorDims: Array[Int] = Array[Int](),
                                       val embedInDims: Array[Int] = Array[Int](),
                                       val embedOutDims: Array[Int] = Array[Int](),
                                       val continuousCols: Array[String] = Array[String](),
                                       val hiddenLayers: Array[Int] = Array(40, 20, 10))
                                      (implicit ev: TensorNumeric[T])
  extends Recommender[T] {

  override def buildModel(): AbstractModule[Tensor[T], Tensor[T], T] = {

    val fullModel = Sequential[T]()

    val wideModel = Sequential().setName("widePart")

    wideModel.add(LookupTableSparse[T](
      wideBaseDims.sum + wideCrossDims.sum, numClasses).setInitMethod(Zeros))
      .add(CAdd(Array(numClasses)))

    // deep model
    val deepModel = Sequential[T]()
    val deepColumn = Concat[T](2)

    // add indicator encoded columns
    var indicatorWidth = 0
    if (indicatorDims.length > 0) {
      indicatorWidth = indicatorDims.sum
      deepColumn.add(Sequential[T]().add(Narrow[T](2, 1, indicatorWidth))).setName("indicator")
    }

    // add embedding columns
    require(embedInDims.length == embedOutDims.length, s"size of embedingColumns should match")
    var embedWidth = 0
    var embedOutputDimSum = 0
    if (embedInDims.length > 0) {
      embedOutputDimSum = embedOutDims.sum
      for (i <- 0 until embedInDims.length) {
        val lookupTable = LookupTable[T](embedInDims(i), embedOutDims(i))
        lookupTable.setWeightsBias(
          Array(Tensor(embedInDims(i), embedOutDims(i)).randn(0, 0.1)))
        deepColumn.add(
          Sequential[T]().add(Select[T](2, indicatorWidth + 1 + embedWidth)).add(lookupTable))
        embedWidth += 1
      }
    }

    // add continuous columns
    var continuousWidth = 0
    if (continuousCols.length > 0) {
      continuousWidth = continuousCols.length
      deepColumn.add(
        Sequential[T]().add(Narrow[T](2, indicatorWidth + embedWidth + 1, continuousWidth)))
    }

    // add hidden layers
    deepModel.add(deepColumn)
      .add(Linear[T](indicatorWidth + embedOutputDimSum + continuousWidth, hiddenLayers(0)))
      .add(ReLU())
    for (i <- 1 to hiddenLayers.length - 1) {
      deepModel.add(Linear[T](hiddenLayers(i - 1), hiddenLayers(i))).add(ReLU())
    }
    deepModel.add(Linear[T](hiddenLayers.last, numClasses))

    // create whole model
    modelType match {
      case "wide_n_deep" =>
        fullModel.add(ParallelTable().add(wideModel).add(deepModel))
          .add(CAddTable()).add(LogSoftMax())
        fullModel.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
      case "wide" =>
        wideModel.add(LogSoftMax())
        wideModel.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
      case "deep" =>
        deepModel.add(LogSoftMax())
        deepModel.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
      case _ =>
        throw new IllegalArgumentException("unknown type")
    }
  }
}

object WideAndDeep {
  def apply[@specialized(Float, Double) T: ClassTag]
  (modelType: String = "wide_n_deep",
   numClasses: Int,
   columnInfo: ColumnFeatureInfo,
   hiddenLayers: Array[Int] = Array(40, 20, 10)
  )(implicit ev: TensorNumeric[T]): WideAndDeep[T] = {
    require(columnInfo.wideBaseCols.length == columnInfo.wideBaseDims.length,
            s"size of wideBaseColumns should match")
    require(columnInfo.wideCrossCols.length == columnInfo.wideCrossDims.length,
            s"size of wideCrossColumns should match")
    require(columnInfo.indicatorCols.length == columnInfo.indicatorDims.length,
            s"size of indicatorColumns should match")
    require(columnInfo.embedCols.length == columnInfo.embedInDims.length &&
            columnInfo.embedCols.length == columnInfo.embedOutDims.length,
            s"size of embedingColumns should match")

    new WideAndDeep[T](modelType,
      numClasses,
      columnInfo.wideBaseDims,
      columnInfo.wideCrossDims,
      columnInfo.indicatorDims,
      columnInfo.embedInDims,
      columnInfo.embedOutDims,
      columnInfo.continuousCols,
      hiddenLayers).build()
  }

  def loadModel[T: ClassTag](path: String,
                             weightPath: String = null)(implicit ev: TensorNumeric[T]):
  WideAndDeep[T] = {
    ZooModel.loadModel(path, weightPath).asInstanceOf[WideAndDeep[T]]
  }
}

