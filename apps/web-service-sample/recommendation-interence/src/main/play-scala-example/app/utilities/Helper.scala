package utilities

import com.fasterxml.jackson.databind.ObjectMapper
import com.intel.analytics.bigdl.dataset.{Sample, TensorSample}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.zoo.models.recommendation.{ColumnFeatureInfo, UserItemFeature}
import ml.combust.mleap.runtime.frame.Transformer
import ml.combust.mleap.runtime.serialization.FrameReader
import models.LoadModel

import scala.collection.immutable.{List, Map}

object Helper extends LoadModel{

  def leapTransform(
                   requestString: String,
                   inputCol: String,
                   outputCol: String,
                   transformer: Transformer,
                   mapper: ObjectMapper
                   ) = {

    val schemaLeap = Map(
      "schema" -> Map(
        "fields" -> List(
          Map("type" -> "string", "name" -> inputCol)
        )
      ),
      "rows" -> List(List(requestString))
    )

    val requestLF = mapper.writeValueAsString(schemaLeap)
    val bytes = requestLF.getBytes("UTF-8")
    val predict = FrameReader("ml.combust.mleap.json").fromBytes(bytes).get
    val frame2 = transformer.transform(predict).get
    val result1 = for (lf <- frame2.select(outputCol)) yield lf.dataset.head(0)
    val result2 = result1.get.asInstanceOf[Double] + 1
    result2
  }

  def revertStringIndex(text: String) = {
    val lookUp = params.skuLookUp.get
    if (lookUp.map(_._2).contains(text)) lookUp.filter(x => {
      x._2 == text
    }).map(_._1).head else "NA"
  }

}
