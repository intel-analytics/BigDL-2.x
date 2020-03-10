package com.intel.analytics.zoo.serving

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.pipeline.inference.InferenceModel
import org.apache.spark.rdd.RDD
import com.intel.analytics.zoo.serving.ClusterServing.{Params, Record}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.util.LongAccumulator


class InferenceStrategy(params: Params,
                        bcModel: Broadcast[InferenceModel],
                        acc: LongAccumulator) extends Serializable {

  /**
   * Use single thread to do inference for one tensor
   * This is used for some BLAS model, such as BigDL
   * Where lower level of parallelism is not supported
   *
   * Throughput will be normal but latency will be high
   */
  def singleThreadInference(r: RDD[(String, Tensor[Float])]): RDD[Record] = {
    r.mapPartitions(it => {
      it.grouped(params.coreNum).flatMap(itemBatch => {
        itemBatch.indices.toParArray.map(i => {
          acc.add(1)
          val uri = itemBatch(i)._1
          val t = itemBatch(i)._2
          val localPartitionModel = bcModel.value
          val result = localPartitionModel.doPredict(t.addSingletonDimension()).toTensor[Float]

          val value = PostProcessing(result, params.filter)

          Record(uri, value)
        })
      })
    })
  }

  /**
   * Use all thread to do inference for one tensor
   * This is used for most of model, such as Tensorflow
   * Where lower level of parallelism is supported
   */
  def allThreadInference(r: RDD[(String, Tensor[Float])]): RDD[Record] = {
    r.mapPartitions(it => {
      val localModel = bcModel.value
      val t = if (params.chwFlag) {
        Tensor[Float](params.coreNum, params.C, params.H, params.W)
      } else {
        Tensor[Float](params.coreNum, params.H, params.W, params.C)
      }
      it.grouped(params.coreNum).flatMap(pathByteBatch => {
        val thisBatchSize = pathByteBatch.size
        acc.add(thisBatchSize)
        (0 until thisBatchSize).toParArray
          .foreach(i => t.select(1, i + 1).copy(pathByteBatch(i)._2))

        val thisT = if (params.chwFlag) {
          t.resize(thisBatchSize, params.C, params.H, params.W)
        } else {
          t.resize(thisBatchSize, params.H, params.W, params.C)
        }
        val x = if (params.modelType == "openvino") {
          thisT.addSingletonDimension()
        } else {
          thisT
        }
        /**
         * addSingletonDimension method will modify the
         * original Tensor, thus if reuse of Tensor is needed,
         * have to squeeze it back.
         */
        val result = if (params.modelType == "openvino") {
          val res = localModel.doPredict(x).toTensor[Float].squeeze()
          t.squeeze(1)
          res
        } else {
          localModel.doPredict(x).toTensor[Float]
        }

        (0 until thisBatchSize).toParArray.map(i => {
          val value = PostProcessing(result.select(1, i + 1), params.filter)
          Record(pathByteBatch(i)._1, value)
        })

      })
    })
  }

}
object InferenceStrategy {
  def apply(params: Params, bcModel:
            Broadcast[InferenceModel],
            acc: LongAccumulator): InferenceStrategy =
    new InferenceStrategy(params, bcModel, acc)
}
