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

package com.intel.analytics.zoo.transform.vision.example

import com.intel.analytics.zoo.transform.vision.image.feature.BGRImageReader
import org.apache.spark.sql.SparkSession

object ImageReaderExample {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("local[1]").appName("test").getOrCreate()

    val imageDF = BGRImageReader.readImagesToBytes(
      "/imagePath/*.jpg",
      spark, 256)

    imageDF.show()

    /**
     *    +--------------------+--------------------+
     *    |                path|           imageData|
     *    +--------------------+--------------------+
     *    |file:/home/yuhao/...|B@26aee0a6,256,...  |
     *    |file:/home/yuhao/...|B@20184ade,256,..  .|
     *    +--------------------+--------------------+
     */

  }

}
