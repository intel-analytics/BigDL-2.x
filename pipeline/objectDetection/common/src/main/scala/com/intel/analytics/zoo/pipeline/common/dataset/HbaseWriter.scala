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

package com.intel.analytics.zoo.pipeline.common.dataset

import java.io.File

import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.zoo.transform.vision.image.ImageFeature
import org.apache.commons.io.FileUtils
import org.apache.hadoop.hbase.client.{ConnectionFactory, Put, Table}
import org.apache.hadoop.hbase.util.Bytes
import org.apache.hadoop.hbase.{HBaseConfiguration, HColumnDescriptor, HTableDescriptor, TableName}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import scopt.OptionParser


object HbaseWriter {
  val logger = Logger.getLogger(getClass)
  val HB_COL_FAMILY = ImageFeature.bytes.getBytes()
  val HB_COL_NAME = ImageFeature.bytes.getBytes

  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics").setLevel(Level.INFO)

  case class HBaseLoaderParam(
    tableName: String = "image",
    imageFolder: String = "",
    overwrite: Boolean = false,
    batch: Int = 64
  )

  val parser = new OptionParser[HBaseLoaderParam]("HBaseLoader") {
    head("HBaseLoader")
    opt[String]('t', "table")
      .text("table name")
      .action((x, c) => c.copy(tableName = x))
    opt[String]('f', "folder")
      .text("image folder")
      .action((x, c) => c.copy(imageFolder = x))
      .required()
    opt[Boolean]("overwrite")
      .text("overrite the table")
      .action((x, c) => c.copy(overwrite = x))
    opt[Int]('b', "batch")
      .text("batch size")
      .action((x, c) => c.copy(batch = x))
  }


  def main(args: Array[String]) {
    parser.parse(args, HBaseLoaderParam()).foreach { params =>
      val configuration = HBaseConfiguration.create()
      val hbaseConn = ConnectionFactory.createConnection(configuration)

      val conf = Engine.createSparkConf().setAppName("HBaseLoader")
      val ss = SparkSession.builder.config(conf)
        .enableHiveSupport().getOrCreate()

      val admin = hbaseConn.getAdmin
      val tableName = TableName.valueOf(params.tableName)

      if (admin.tableExists(tableName) && params.overwrite) {
        admin.disableTable(tableName)
        admin.deleteTable(tableName)
        ss.sql(s"DROP TABLE IF EXISTS ${params.tableName}")
        logger.info(s"delete table $tableName")
      }
      if (!admin.tableExists(tableName)) {
        // Instantiating table descriptor class
        val tableDescriptor = new HTableDescriptor(tableName)
        // Adding column families to table descriptor
        tableDescriptor.addFamily(new HColumnDescriptor(HB_COL_FAMILY))
        // Execute the table through admin
        admin.createTable(tableDescriptor)
        logger.info(" Table created ")
      }

      val images = new File(params.imageFolder).listFiles()
      val groupedImages = images.grouped(params.batch)

      val table = hbaseConn.getTable(TableName.valueOf(params.tableName))

      groupedImages.foreach(images => {
        putImages(table, images)
      })
      ss.close()
      hbaseConn.close()
    }
  }

  import scala.collection.JavaConverters._

  def putImages(table: Table, images: Array[File]): Unit = {
    val puts = images.map(path => {
      val image = FileUtils.readFileToByteArray(path)
      val put = new Put(Bytes.toBytes(path.toString))
      put.addImmutable(HB_COL_FAMILY, HB_COL_NAME, image)
      put
    }).toList.asJava
    table.put(puts)
  }
}

