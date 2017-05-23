package com.intel.analytics.zoo.pipeline.deepspeech2.pipeline.acoustic


import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, Row}


class TimeSegmenter ( override val uid: String)
  extends Transformer with HasInputCol with HasOutputCol with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("TimeSegmenter"))

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)


  val segmentSize = new IntParam(this, "segmentSize", "segmentSize",
    ParamValidators.gt(0))

  /** @group getParam */
  def getSegmentSize: Int = $(segmentSize)

  /** @group setParam */
  def setSegmentSize(value: Int): this.type = set(segmentSize, value)

  setDefault(segmentSize -> 16000 * 30)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val outputSchema = transformSchema(dataset.schema)

    val rows = dataset.select("path", "target", $(inputCol)).rdd.zipWithIndex().flatMap { case (r, id) =>
      val path = r.getAs[String](0)
      val target = r.getAs[String](1)
      val arr = r.getSeq[Float](2).toArray.grouped($(segmentSize))
      arr.zipWithIndex.map { case (data, seq) =>
        Row(path, target, id, seq, data)
      }
    }
    val schema = StructType(Seq(
      StructField("path", StringType, nullable = false),
      StructField("target", StringType, nullable = false),
      StructField("audio_id", LongType, nullable = false),
      StructField("audio_seq", IntegerType, nullable = false),
      StructField($(outputCol), dataset.schema($(inputCol)).dataType, nullable = false)
    ))
    dataset.sparkSession.createDataFrame(rows, schema)
  }

  override def transformSchema(schema: StructType): StructType = {
    require(!schema.fieldNames.contains($(outputCol)),
      s"Output column ${$(outputCol)} already exists.")
    val outputFields = schema.fields :+ StructField($(outputCol), schema($(inputCol)).dataType, false)
    StructType(outputFields)
  }

  override def copy(extra: ParamMap): TimeSegmenter = defaultCopy(extra)
}


object TimeSegmenter extends DefaultParamsReadable[TimeSegmenter] {

  override def load(path: String): TimeSegmenter = super.load(path)
}
