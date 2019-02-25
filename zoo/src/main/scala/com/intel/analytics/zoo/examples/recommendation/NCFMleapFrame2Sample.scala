package com.intel.analytics.zoo.examples.recommendation

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.zoo.common.MleapFrame2Sample
import com.intel.analytics.zoo.models.recommendation.UserItemFeature
import ml.combust.mleap.runtime.DefaultLeapFrame
import org.apache.spark.rdd.RDD

class NCFMleapFrame2Sample extends MleapFrame2Sample {
  override def toSample(frame: DefaultLeapFrame): Array[Sample[Float]] = {

     frame.select("userId", "itemId", "label")
       .get
       .dataset
       .map(row => {
      val uid = row.getAs[Int](0) + 1
      val iid = row.getAs[Int](1) + 1

      val label = row.getAs[Int](2)
      val feature: Tensor[Float] = Tensor[Float](T(uid.toFloat, iid.toFloat))

      Sample(feature, Tensor[Float](T(label)))
    }).toArray
  }
}
