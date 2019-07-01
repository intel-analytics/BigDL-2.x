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

import java.lang

import com.intel.analytics.bigdl.dataset.{Sample, TensorSample}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import org.apache.spark.sql.functions.{max, udf}
import org.apache.spark.sql.{DataFrame, Row}

import scala.collection.mutable
import scala.util.Random
import scala.collection.JavaConverters._

object Utils {

  /**
   * generate negative samples given a dataframe of positive records, label >=2.
   *
   * @param indexed dataframe positive of userId, itemId and label.
   * @return a dataframe of negative samples(label=1) with the same size as indexed dataframe
   */
  def getNegativeSamples(indexed: DataFrame): DataFrame = {
    val schema = indexed.schema
    require(schema.fieldNames.contains("userId"), s"Column userId should exist")
    require(schema.fieldNames.contains("itemId"), s"Column itemId should exist")
    require(schema.fieldNames.contains("label"), s"Column label should exist")

    val indexedDF = indexed.select("userId", "itemId", "label")
    val minMaxRow = indexedDF.agg(max("userId"), max("itemId")).collect()(0)
    val (userCount, itemCount) = (minMaxRow.getInt(0), minMaxRow.getInt(1))
    val sampleDict = indexedDF.rdd.map(row => row(0) + "," + row(1)).collect().toSet

    val dfCount = indexedDF.count.toInt

    import indexed.sqlContext.implicits._

    @transient lazy val ran = new Random(System.nanoTime())

    val negative = indexedDF.rdd
      .map(x => {
        val uid = x.getAs[Int](0)
        val iid = Math.max(ran.nextInt(itemCount), 1)
        (uid, iid)
      })
      .filter(x => !sampleDict.contains(x._1 + "," + x._2)).distinct()
      .map(x => (x._1, x._2, 1))
      .toDF("userId", "itemId", "label")

    negative
  }

  def buckBucket(bucketSize: Int): (String, String) => Int = {
    val func = (col1: String, col2: String) =>
      (Math.abs((col1 + "_" + col2).hashCode()) % bucketSize + 0)
    func
  }

  def buckBuckets(bucketSize: Int)(col: String*): Int = {
    Math.abs(col.reduce(_ + "_" + _).hashCode()) % bucketSize + 0
  }

  def bucketizedColumn(boundaries: Array[Float]): Float => Int = {
    col1: Float => {
      var index = 0
      while (index < boundaries.length && col1 >= boundaries(index)) {
        index += 1
      }
      index
    }
  }

  // save 0 for uncovered ones
  def categoricalFromVocabList(vocabList: Array[String]): (String) => Int = {
    val func = (sth: String) => {
      val default: Int = 0
      val start: Int = 1
      if (vocabList.contains(sth)) vocabList.indexOf(sth) + start
      else default
    }
    func
  }


  /**
   * convert a row to sample given column information of WideAndDeep model.
   *
   * @param r          Row of userId, itemId, features and label
   * @param columnInfo ColumnFeatureInfo specify information of different features
   * @param modelType  support "wide_n_deep", "wide", "deep" only
   * @return TensorSample as input for WideAndDeep model
   */
  def row2Sample(r: Row, columnInfo: ColumnFeatureInfo, modelType: String): Sample[Float] = {

    val wideTensor: Tensor[Float] = getWideTensor(r, columnInfo)
    val deepTensor: Array[Tensor[Float]] = getDeepTensors(r, columnInfo)
    val l = r.getAs[Int](columnInfo.label)
    val label = Tensor[Float](T(l))
    label.resize(1, 1)

    modelType match {
      case "wide_n_deep" =>
        TensorSample[Float](Array(wideTensor) ++ deepTensor, Array(label))
      case "wide" =>
        TensorSample[Float](Array(wideTensor), Array(label))
      case "deep" =>
        TensorSample[Float](deepTensor, Array(label))
      case _ =>
        throw new IllegalArgumentException("unknown type")
    }
  }

  /**
   * convert a row to sample given column information of WideAndDeep Sequential model.
   *
   * @param r Row of userId, itemId, features and label
   * @param columnInfo ColumnFeatureInfo specify information of different features
   * @param modelType support "wide_n_deep", "wide", "deep" only
   * @return TensorSample as input for WideAndDeep Sequential model
   */
  def row2SampleSequential(r: Row, columnInfo: ColumnFeatureInfo, modelType: String): Sample[Float]
  = {
    val wideTensor: Tensor[Float] = getWideTensor(r, columnInfo)
    val deepTensor: Tensor[Float] = getDeepTensor(r, columnInfo)
    val l = r.getAs[Int](columnInfo.label)

    val label = Tensor[Float](T(l))
    label.resize(1, 1)

    modelType match {
      case "wide_n_deep" =>
        TensorSample[Float](Array(wideTensor, deepTensor), Array(label))
      case "wide" =>
        TensorSample[Float](Array(wideTensor), Array(label))
      case "deep" =>
        TensorSample[Float](Array(deepTensor), Array(label))
      case _ =>
        throw new IllegalArgumentException("unknown type")
    }
  }


  /**
   * convert a row to tensor given column feature information of WideAndDeep model.
   *
   * @param r          Row of userId, itemId, features and label
   * @param columnInfo ColumnFeatureInfo specify information of different features
   * @return a tensor as input for wide part of a WideAndDeep model
   */
  def getWideTensor(r: Row, columnInfo: ColumnFeatureInfo): Tensor[Float] = {
    val wideColumns = columnInfo.wideBaseCols ++ columnInfo.wideCrossCols
    val wideDims = columnInfo.wideBaseDims ++ columnInfo.wideCrossDims
    val wideLength = wideColumns.length
    var acc = 0
    val indices: Array[Int] = (0 to wideLength - 1).map(i => {
      val index = r.getAs[Int](wideColumns(i))
      if (i == 0) index
      else {
        acc = acc + wideDims(i - 1)
        acc + index
      }
    }).toArray
    val values = indices.map(_ + 1.0f)
    val shape = Array(wideDims.sum)

    Tensor.sparse(Array(indices), values, shape)
  }

  /**
   * convert a row to tensors given column feature information of WideAndDeep model.
   *
   * @param r          Row of userId, itemId, features and label
   * @param columnInfo ColumnFeatureInfo specify information of different features
   * @return an array of tensors as input for deep part of a WideAndDeep model
   */
  def getDeepTensors(r: Row, columnInfo: ColumnFeatureInfo): Array[Tensor[Float]] = {

    val indCol = columnInfo.indicatorCols.length
    val embCol = columnInfo.embedCols.length
    val contCol = columnInfo.continuousCols.length

    val indTensor = Tensor[Float](columnInfo.indicatorDims.sum).fill(0)

    // setup indicators
    (0 to indCol - 1).map {
      i =>
        val index = r.getAs[Int](columnInfo.indicatorCols(i))
        val accIndex = if (i == 0) {
          index
        }
        else {
          columnInfo.indicatorDims(i - 1)
        }
        indTensor.setValue(accIndex + 1, 1)
    }

    val embTensor = Tensor[Float](embCol).fill(0)
    (0 to embCol - 1).map(i =>
      embTensor.setValue(i + 1, r.getAs[Int](columnInfo.embedCols(i)).toFloat))


    val contTensor = Tensor[Float](contCol).fill(0)
    (0 to contCol - 1).map(i =>
      contTensor.setValue(i + 1, r.getAs[Int](columnInfo.continuousCols(i)).toFloat))

    (indCol > 0, embCol > 0, contCol > 0) match {

      case (true, true, true) =>
        Array(indTensor, embTensor, contTensor)
      case (false, true, true) =>
        Array(embTensor, contTensor)
      case (true, false, true) =>
        Array(indTensor, contTensor)
      case (true, true, false) =>
        Array(indTensor, embTensor)
      case (false, true, false) =>
        Array(embTensor)
      case (false, false, true) =>
        Array(contTensor)
      case (true, false, false) =>
        Array(indTensor)
      case _ =>
        Array[Tensor[Float]]()
    }

  }

  // setup deep tensor
  def getDeepTensor(r: Row, columnInfo: ColumnFeatureInfo): Tensor[Float] = {
    val deepColumns1 = columnInfo.indicatorCols
    val deepColumns2 = columnInfo.embedCols ++ columnInfo.continuousCols
    val deepLength = columnInfo.indicatorDims.sum + deepColumns2.length
    val deepTensor = Tensor[Float](deepLength).fill(0)

    // setup indicators
    var acc = 0
    (0 to deepColumns1.length - 1).map {
      i =>
        val index = r.getAs[Int](columnInfo.indicatorCols(i))
        val accIndex = if (i == 0) index
        else {
          acc = acc + columnInfo.indicatorDims(i - 1)
          acc + index
        }
        deepTensor.setValue(accIndex + 1, 1)
    }

    // setup embedding and continuous
    (0 to deepColumns2.length - 1).map {
      i =>
        deepTensor.setValue(i + 1 + columnInfo.indicatorDims.sum,
          r.getAs[Int](deepColumns2(i)).toFloat)
    }
    deepTensor
  }

  def rows2sample(r: Row,
                  sessionLength: Int,
                  includeHistory: Boolean,
                  historyLength: Int): Sample[Float] = {
    val label = Tensor[Float](T(r.getAs[Float]("label")))
    val rnnFeature: Array[Float] = r
      .getAs[mutable.WrappedArray[java.lang.Float]]("session").array.map(_.toFloat)
    val rnnTensor = Tensor(rnnFeature, Array(sessionLength))

    val sample = if (includeHistory) {
      val mlpFeature: Array[Float] = r
        .getAs[mutable.WrappedArray[java.lang.Float]]("purchase_history").array.map(_.toFloat)
      val mlpTensor = Tensor(mlpFeature, Array(historyLength))
      Sample[Float](Array(mlpTensor, rnnTensor), Array(label))
    }
    else {
      Sample[Float](Array(rnnTensor), Array(label))
    }
    sample
  }

  def prePadding(maxLength: Int): mutable.WrappedArray[java.lang.Float] => Array[Float] = {

    (seq: mutable.WrappedArray[java.lang.Float]) => {
      if (seq.array.size < maxLength) {
        seq.array.map(_.toFloat).reverse.padTo(maxLength, 0f).reverse
      }
      else {
        seq.array.map(_.toFloat).takeRight(maxLength)
      }
    }
  }

  def slideSession(df: DataFrame, sessionLength: Int): DataFrame = {
    val sqlContext = df.sqlContext
    import sqlContext.implicits._

    val dfSlided = df.rdd.flatMap(x => {
      val session: Array[Float] = x.getAs[mutable.WrappedArray[java.lang.Float]]("session")
        .array.map(_.toFloat)
      val feature2 = x.getAs[mutable.WrappedArray[java.lang.Float]]("purchase_history")
        .array.map(_.toFloat)
      val featureLabel = for (label <- session.slice(1, session.size)) yield {
        val endIdx = session.indexOf(label)
        val beginIdx = if (session.size <= sessionLength) 0 else endIdx - sessionLength
        val feature1 = session.slice(beginIdx, endIdx)
        (feature1, feature2, label)
      }
      featureLabel
    }).toDF("session", "purchase_history", "label").na.drop()
    dfSlided
  }

}
