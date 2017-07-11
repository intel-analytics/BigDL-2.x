/*
 * Copyright 2016 The BigDL Authors.
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

package com.intel.analytics.zoo.pipeline.deepspeech2.pipeline.acoustic

import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.ml.param.{Param, ParamMap, ParamValidators}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Dataset, Row}

/**
 * Evaluator for Speech Recognition output.
 */
final class ASREvaluator (override val uid: String) extends Evaluator
  with HasPredictionCol with HasLabelCol with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("ASREvaluator"))

  val metricName: Param[String] = {
    val allowedParams = ParamValidators.inArray(Array("wer", "cer"))
    new Param(this, "metricName", "metric name in evaluation (wer|cer)", allowedParams)
  }
  setDefault(metricName -> "cer")

  /** @group getParam */
  def getMetricName: String = $(metricName)

  /** @group setParam */
  def setMetricName(value: String): this.type = set(metricName, value)

  /** @group setParam */
  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  /** @group setParam */
  def setLabelCol(value: String): this.type = set(labelCol, value)

  override def evaluate(dataset: Dataset[_]): Double = {
    val schema = dataset.schema
    SchemaUtils.checkColumnType(schema, $(predictionCol), StringType)
    SchemaUtils.checkColumnType(schema, $(labelCol), StringType)

    if ($(metricName) == "cer") {
      val predictionAndLabels = dataset
        .select(col($(predictionCol)).cast(StringType), col($(labelCol)).cast(StringType)).rdd
        .map { case Row(prediction: String, label: String) =>
          ((cer(prediction, label) * label.length).toLong, label.length.toLong)
        }
      val err = predictionAndLabels.map(_._1).sum()
      val total = predictionAndLabels.map(_._2).sum()
      err / total
    }
    else {
      val predictionAndLabels = dataset
        .select(col($(predictionCol)).cast(StringType), col($(labelCol)).cast(StringType))
        .rdd
        .map { case Row(prediction: String, label: String) =>
          ((wer(prediction, label) * label.split(" ").length).toLong, label.split(" ").length.toLong)
        }
      val err = predictionAndLabels.map(_._1).sum()
      val total = predictionAndLabels.map(_._2).sum()
      err / total
    }
  }

  override def isLargerBetter: Boolean = $(metricName) match {
    case "cer" => false
    case "wer" => false
  }

  private def stringDistance(s1: String, s2: String): Int = {
    def sd(s1: List[Char], s2: List[Char], costs: List[Int]): Int = s2 match {
      case Nil => costs.last
      case c2 :: tail => sd( s1, tail,
        (List(costs.head+1) /: costs.zip(costs.tail).zip(s1))((a,b) => b match {
          case ((rep,ins), chr) => Math.min( Math.min( ins+1, a.head+1 ), rep + (if (chr==c2) 0 else 1) ) :: a
        }).reverse
      )
    }
    sd(s1.toList, s2.toList, (0 to s1.length).toList)
  }

  def cer(output: String, target: String): Double = {
    stringDistance(output, target).toDouble / target.length
  }

  def wer(output: String, target: String): Double = {
    val b = (output.split(" ") ++ target.split(" ")).toSet
    val word2char = b.zipWithIndex.toMap
    val w1 = output.split(" ").map(s => word2char(s).toChar)
    val w2 = target.split(" ").map(s => word2char(s).toChar)
    stringDistance(w1.mkString(""), w2.mkString("")).toDouble / target.split(" ").length
  }

  override def copy(extra: ParamMap): ASREvaluator = defaultCopy(extra)
}


object RegressionEvaluator extends DefaultParamsReadable[ASREvaluator] {

  override def load(path: String): ASREvaluator = super.load(path)
}


