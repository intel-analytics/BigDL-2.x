package com.intel.analytics.zoo.models.recommendation

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.models.common.ZooModel
import com.intel.analytics.zoo.pipeline.api.keras.layers.{Dense, Embedding, GRU}
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential

import scala.reflect.ClassTag

class SessionRecommender[T: ClassTag](val itemCnt: Int,
                                      val embedDim: Int,
                                      val featureLength: Int,
                                      val hiddenUnit: Int = 100
                                     )(implicit ev: TensorNumeric[T])
  extends Recommender[T] {

  override def buildModel(): AbstractModule[Tensor[T], Tensor[T], T] = {
    val model = Sequential[Float]()

    model.add(Embedding[Float](itemCnt, embedDim, init = "normal", inputLength = featureLength))
      .add(GRU[Float](hiddenUnit, returnSequences = false))
      .add(Dense[Float](itemCnt, activation = "log_softmax"))
    model.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }

}

object SessionRecommender {

  def apply[@specialized(Float, Double) T: ClassTag](itemCnt: Int,
                                                     embedDim: Int,
                                                     featureLength: Int,
                                                     hiddenUnit: Int = 100)(implicit ev: TensorNumeric[T]): SessionRecommender[T] =
    new SessionRecommender[T](itemCnt, embedDim, featureLength, hiddenUnit).build()

  def loadModel[T: ClassTag](path: String,
                             weightPath: String = null)(implicit ev: TensorNumeric[T]): SessionRecommender[T] = {
    ZooModel.loadModel(path, weightPath).asInstanceOf[SessionRecommender[T]]
  }
}
