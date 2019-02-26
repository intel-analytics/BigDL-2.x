package com.intel.analytics.zoo.models.recommendation

import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.zoo.pipeline.api.keras.models.Model
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.models.common.ZooModel
import com.intel.analytics.zoo.pipeline.api.keras.layers._

import scala.reflect.ClassTag

/**
  * The factory method to create a SessionRecommender instance.
  *
  * @param itemCount The number of distinct items. Positive integer.
  * @param numClasses The number of classes. Positive integer.
  * @param itemEmbed The output size of embedding layer. Positive integer.
  * @param mlpHiddenLayers Units of hidden layers for the mlp model. Array of positive integers. Default is Array(40, 20, 10).
  * @param rnnHiddenLayers Units of hidden layers for the mlp model. Array of positive integers. Default is Array(200, 200).
  * @param includeHistory Whether to include purchase history. Boolean. Default is true.
  * @param maxLength The max number of tokens
  */
class SessionRecommender[T: ClassTag](
                                       val itemCount: Int,
                                       val numClasses: Int,
                                       val itemEmbed: Int = 300,
                                       val mlpHiddenLayers: Array[Int] = Array(40, 20, 10),
                                       val rnnHiddenLayers: Array[Int] = Array(200, 200),
                                       val includeHistory: Boolean = true,
                                       val maxLength: Int = 10
                                     )(implicit ev: TensorNumeric[T])
  extends Recommender[T] {

  override def buildModel(): AbstractModule[Tensor[T], Tensor[T], T] = {

    // input variables
    val inputRNN: ModuleNode[T] = Input(inputShape = Shape(maxLength))

    // item embedding layer
    val itemTable = Embedding[T](itemCount + 1, itemEmbed, inputLength = maxLength)

    // rnn part
    var gru = GRU[T](rnnHiddenLayers(0), returnSequences = true).inputs(itemTable.inputs(inputRNN))
    for (i <- 1 until rnnHiddenLayers.length - 1) {
      gru = GRU[T](rnnHiddenLayers(i), returnSequences = true).inputs(gru)
    }
    val gruLast = GRU[T](rnnHiddenLayers.last, returnSequences = false).inputs(gru)
    val rnn = Dense[T](numClasses, activation = "log_softmax").inputs(gruLast)

    if (includeHistory) {
      // mlp part
      val inputMLP: ModuleNode[T] = Input(inputShape = Shape(itemCount))
      var mlp = Dense(mlpHiddenLayers(0)).inputs(itemTable.inputs(inputMLP))
      for (i <- 1 until mlpHiddenLayers.length) {
        mlp = Dense(mlpHiddenLayers(i), activation = "relu").inputs(mlp)
      }
      // combine rnn and mlp parts
      val model = Merge.merge[T](List(mlp, rnn), "sum")
      Model(Array(inputMLP, inputRNN), model).asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
    }
    else Model(inputRNN, rnn).asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object SessionRecommender {
  /**
    * The factory method to create a SessionRecommender instance.
    *
    * @param itemCount The number of distinct items. Positive integer.
    * @param numClasses The number of classes. Positive integer.
    * @param itemEmbed The output size of embedding layer. Positive integer.
    * @param mlpHiddenLayers Units of hidden layers for the deep model. Array of positive integers.
    *                     Default is Array(40, 20, 10).
    * @param includeHistory Whether to include purchase history. Boolean. Default is true.
    * @param maxLength The max number of tokens
    */
  def apply[@specialized(Float, Double) T: ClassTag](
                                                      itemCount: Int,
                                                      numClasses: Int,
                                                      itemEmbed: Int,
                                                      mlpHiddenLayers: Array[Int] = Array(200, 100, 100),
                                                      rnnHiddenLayer: Array[Int] = Array(200, 200),
                                                      includeHistory: Boolean = true,
                                                      maxLength: Int = 10)
                                                    (implicit ev: TensorNumeric[T]): SessionRecommender[T] = {
    new SessionRecommender[T](
      itemCount,
      numClasses,
      itemEmbed,
      mlpHiddenLayers,
      rnnHiddenLayer,
      includeHistory,
      maxLength
    ).build()
  }

  /**
    * Load an existing SeqRecommender model (with weights).
    *
    * @param path The path for the pre-defined model.
    *             Local file system, HDFS and Amazon S3 are supported.
    *             HDFS path should be like "hdfs://[host]:[port]/xxx".
    *             Amazon S3 path should be like "s3a://bucket/xxx".
    * @param weightPath The path for pre-trained weights if any. Default is null.
    * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
    */
  def loadModel[T: ClassTag](
                              path: String,
                              weightPath: String = null
                            )(implicit ev: TensorNumeric[T]): SessionRecommender[T] = {
    ZooModel.loadModel(path, weightPath).asInstanceOf[SessionRecommender[T]]
  }

}
