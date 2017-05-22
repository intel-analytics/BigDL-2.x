package com.intel.analytics.deepspeech2.pipeline.acoustic

import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector}
import org.apache.spark.ml.param.{Param, Params}
import org.apache.spark.sql.types.{DataType, NumericType, StructField, StructType}

/**
  * Created by yuhao on 3/9/17.
  */
private[pipeline] trait HasInputCol extends Params {

  /**
    * Param for input column name.
    * @group param
    */
  final val inputCol: Param[String] = new Param[String](this, "inputCol", "input column name")

  /** @group getParam */
  final def getInputCol: String = $(inputCol)
}

private[pipeline] trait HasOutputCol extends Params {

  /**
    * Param for output column name.
    * @group param
    */
  final val outputCol: Param[String] = new Param[String](this, "outputCol", "output column name")

  setDefault(outputCol, uid + "__output")

  /** @group getParam */
  final def getOutputCol: String = $(outputCol)
}


/**
  * Trait for shared param labelCol (default: "label").
  */
private[pipeline] trait HasLabelCol extends Params {

  /**
    * Param for label column name.
    * @group param
    */
  final val labelCol: Param[String] = new Param[String](this, "labelCol", "label column name")

  setDefault(labelCol, "label")

  /** @group getParam */
  final def getLabelCol: String = $(labelCol)
}

/**
 * Trait for shared param windowCol (default: "window").
 */
private[pipeline] trait HasWindowCol extends Params {

  /**
   * Param for label column name.
   * @group param
   */
  final val windowCol: Param[String] = new Param[String](this, "windowCol", "window column name")

  setDefault(windowCol, "window")

  /** @group getParam */
  final def getWindowCol: String = $(windowCol)
}

/**
  * Trait for shared param predictionCol (default: "prediction").
  */
private[pipeline] trait HasPredictionCol extends Params {

  /**
    * Param for prediction column name.
    * @group param
    */
  final val predictionCol: Param[String] = new Param[String](this, "predictionCol", "prediction column name")

  setDefault(predictionCol, "prediction")

  /** @group getParam */
  final def getPredictionCol: String = $(predictionCol)
}

/**
 * Trait for shared param blockSize.
 */
private[pipeline] trait HasBlockSize extends Params {

  /**
   * Block size for stacking input data in matrices to speed up the computation.
   * Data is stacked within partitions. If block size is more than remaining data in
   * a partition then it is adjusted to the size of this data.
   * Recommended size is between 10 and 1000.
   * Default: 128
   *
   * @group expertParam
   */
  final val blockSize: Param[Int] = new Param[Int](this, "blockSize",
    "Block size for stacking input data in matrices. Data is stacked within partitions." +
      " If block size is more than remaining data in a partition then " +
      "it is adjusted to the size of this data. Recommended size is between 10 and 1000")

  setDefault(blockSize, 0)

  /** @group expertGetParam */
  final def getBlockSize: Int = $(blockSize)
}

/**
 * Trait for shared param maxIter.
 */
private[pipeline] trait HasMaxIter extends Params {

  /**
   * Param for maximum number of iterations (>= 0).
   * @group param
   */
  final val maxIter: Param[Int] = new Param[Int](this, "maxIter", "maximum number of iterations (>= 0)")

  setDefault(maxIter, 0)

  /** @group getParam */
  final def getMaxIter: Int = $(maxIter)
}

/**
  * Utils for handling schemas.
  */
private[pipeline] object SchemaUtils {

  // TODO: Move the utility methods to SQL.

  /**
    * Check whether the given schema contains a column of the required data type.
    * @param colName  column name
    * @param dataType  required column data type
    */
  def checkColumnType(
                       schema: StructType,
                       colName: String,
                       dataType: DataType,
                       msg: String = ""): Unit = {
    val actualDataType = schema(colName).dataType
    val message = if (msg != null && msg.trim.length > 0) " " + msg else ""
    require(actualDataType.equals(dataType),
      s"Column $colName must be of type $dataType but was actually $actualDataType.$message")
  }

  /**
    * Check whether the given schema contains a column of one of the require data types.
    * @param colName  column name
    * @param dataTypes  required column data types
    */
  def checkColumnTypes(
                        schema: StructType,
                        colName: String,
                        dataTypes: Seq[DataType],
                        msg: String = ""): Unit = {
    val actualDataType = schema(colName).dataType
    val message = if (msg != null && msg.trim.length > 0) " " + msg else ""
    require(dataTypes.exists(actualDataType.equals),
      s"Column $colName must be of type equal to one of the following types: " +
        s"${dataTypes.mkString("[", ", ", "]")} but was actually of type $actualDataType.$message")
  }

  /**
    * Check whether the given schema contains a column of the numeric data type.
    * @param colName  column name
    */
  def checkNumericType(
                        schema: StructType,
                        colName: String,
                        msg: String = ""): Unit = {
    val actualDataType = schema(colName).dataType
    val message = if (msg != null && msg.trim.length > 0) " " + msg else ""
    require(actualDataType.isInstanceOf[NumericType], s"Column $colName must be of type " +
      s"NumericType but was actually of type $actualDataType.$message")
  }

  /**
    * Appends a new column to the input schema. This fails if the given output column already exists.
    * @param schema input schema
    * @param colName new column name. If this column name is an empty string "", this method returns
    *                the input schema unchanged. This allows users to disable output columns.
    * @param dataType new column data type
    * @return new schema with the input column appended
    */
  def appendColumn(
                    schema: StructType,
                    colName: String,
                    dataType: DataType,
                    nullable: Boolean = false): StructType = {
    if (colName.isEmpty) return schema
    appendColumn(schema, StructField(colName, dataType, nullable))
  }

  /**
    * Appends a new column to the input schema. This fails if the given output column already exists.
    * @param schema input schema
    * @param col New column schema
    * @return new schema with the input column appended
    */
  def appendColumn(schema: StructType, col: StructField): StructType = {
    require(!schema.fieldNames.contains(col.name), s"Column ${col.name} already exists.")
    StructType(schema.fields :+ col)
  }
}


/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

