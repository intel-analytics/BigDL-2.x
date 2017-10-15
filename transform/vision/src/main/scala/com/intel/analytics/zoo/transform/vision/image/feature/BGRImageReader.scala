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

package com.intel.analytics.zoo.transform.vision.image.feature

import java.io.Serializable

import com.intel.analytics.zoo.transform.vision.image.feature.ImageData.BytesImage
import com.intel.analytics.zoo.transform.vision.image.opencv.OpenCV
import org.apache.commons.io.IOUtils
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.param.{FloatParam, IntParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Encoders, Row, SparkSession}

object ImageData {
  type BytesImage = (Array[Byte], Int, Int, Int)
  type FloatsImage = (Array[Float], Int, Int, Int)
}

/**
 * BGRImageReader extends Spark Transformer and provides function for loading image from
 * local or HDFS paths.
 *
 * When used as a Transformer, the input column should be of string type, and contains a local
 * or HDFS path. The output column contains a tuple of imageData (data: Array[Bytes], height: Int,
 * width: Int, numChannel: Int). The data is stored in a pixel-by-pixel BGR row-wise order.
 * This follows the OpenCV convention.
 *
 * Object BGRImageReader also provides static function for loading all the images in a
 * specific path, where the path support wildcard matching, e.g. /path/1*.jpg
 *
 * In the case of reading failure (image file malformed), the output column presents null.
 */
class BGRImageReader(override val uid: String)
  extends UnaryTransformer[String, BytesImage, BGRImageReader] {

  def this() = this(Identifiable.randomUID("BGRImageReader"))

  final val smallSideSize: IntParam = new IntParam(this, "scaleTo", "scaleTo")

  def getSmallSideSize: Int = $(smallSideSize)

  def setSmallSideSize(value: Int): this.type = set(smallSideSize, value)
  setDefault(smallSideSize -> 256)

  override protected def createTransformFunc: String => BytesImage = (path: String) => {
    try {
      val src: Path = new Path(path)
      val fs = src.getFileSystem(new Configuration())
      val is = fs.open(src)
      val fileBytes = IOUtils.toByteArray(is)
      ReadImageUtil.readImageAsBytes(fileBytes, $(smallSideSize))
    }
    catch {
      case e: Exception =>
        logWarning("ERROR: error when reading " + path)
        null
    }
  }

  override protected def outputDataType: DataType =
    new StructType()
      .add(StructField("_1", BinaryType))
      .add(StructField("_2", IntegerType))
      .add(StructField("_3", IntegerType))
      .add(StructField("_4", IntegerType))

  override protected def validateInputType(inputType: DataType): Unit = {
    require(inputType == DataTypes.StringType, s"Bad input type: $inputType. Requires String.")
  }

}

/**
 * Object BGRImageReader provides static function for loading all the images by a
 * specific path, where the path support wildcard matching, e.g. /path/1*.jpg
 *
 * In the case of reading failure (image file malformed), the output column presents null.
 */
object BGRImageReader extends Serializable {

  OpenCV.loadIfNecessary()

  /**
   * read image from local file system or HDFS, resize to specific size.
   * @param path path to read images. Local or HDFS. Wildcard character are supported.
   * @param smallSideSize the size of the smallest side after resize
   * @return a DataFrame contains two columns.
   *         DataFrame("path"[String], "imageData"[Array[Bytes], height:Int, width:Int,
   *         numChannel:Int])
   */
  def readImagesToBytes(path: String,
      spark: SparkSession,
      smallSideSize: Int): DataFrame = {

    import spark.implicits._
    val images = spark.sparkContext.binaryFiles(path)
      .map { case (p, stream) =>
        val fileBytes = stream.toArray()
        val (bytes, h, w, c) = ReadImageUtil.readImageAsBytes(fileBytes, smallSideSize)
        (p, (bytes, h, w, c))
      }
    images.toDF("path", "imageData")
  }

  /**
   * read image from local file system or HDFS
   * @param path path to read images. Local or HDFS. Wildcard character are supported.
   * @return a DataFrame contains two columns.
   *         DataFrame("path"[String], "imageData"[Array[Bytes], height:Int, width:Int,
   *         numChannel:Int])
   */
  def readImagesToBytes(path: String,
      spark: SparkSession): DataFrame = {

    import spark.implicits._
    val images = spark.sparkContext.binaryFiles(path)
      .map { case (p, stream) =>
        val fileBytes = stream.toArray()
        val (bytes, h, w, c) = ReadImageUtil.readImageAsBytes(fileBytes)
        (p, (bytes, h, w, c))
      }
    images.toDF("path", "imageData")
  }

  /**
   * read image from local file system or HDFS, rescale and normalize to the specific range.
   * @param path path to read images. Local or HDFS. Wildcard character are supported.
   * @param smallSideSize specify the shorter dimension after image scaling.
   * @param divisor divide each pixel by divisor. E.g. if divisor = 255f, each pixel is in [0, 1]
   * @return a DataFrame contains two columns.
   *         DataFrame("path"[String], "imageData"[Array[Float], height:Int, width:Int,
   *         numChannel:Int])
   */
  def readImagesAsFloats(
      path: String,
      spark: SparkSession,
      smallSideSize: Int = 256,
      divisor: Float = 1.0f): DataFrame = {

    import spark.implicits._
    val images = spark.sparkContext.binaryFiles(path)
      .map { case (p, stream) =>
        val bytes = stream.toArray()
        val (floats, h, w, c) = ReadImageUtil.readImageAsFloats(bytes, smallSideSize, divisor)
        (p, (floats, h, w, c))
      }
    images.toDF("path", "imageData")
  }

}
