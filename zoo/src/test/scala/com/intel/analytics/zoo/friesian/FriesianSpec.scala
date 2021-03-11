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

package com.intel.analytics.zoo.friesian

import java.net.URL

import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import com.intel.analytics.zoo.friesian.python.PythonFriesian
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SQLContext
import scala.collection.JavaConverters._

class FriesianSpec extends ZooSpecHelper {
  var sqlContext: SQLContext = _
  var sc: SparkContext = _

  override def doBefore(): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = new SparkConf().setMaster("local[1]").setAppName("NCFTest")
    sc = NNContext.initNNContext(conf)
    sqlContext = SQLContext.getOrCreate(sc)
    val resource: URL = getClass.getClassLoader.getResource("recommender")
  }

  "Fill NA" should "work properly" in {
    val resource: URL = getClass.getClassLoader.getResource("friesian")
    val path = resource.getFile + "/data1.parquet"
    val df = sqlContext.read.parquet(path)
    val friesian = PythonFriesian.ofFloat()
    val cols = Array("col_1")
    val dfFilled = friesian.fillNA(df, 0, cols.toList.asJava)
    dfFilled.show()
  }
}
