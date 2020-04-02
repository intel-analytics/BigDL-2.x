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

package com.intel.analytics.zoo.serving

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.pipeline.inference.InferenceModel
import org.apache.spark.rdd.RDD
import com.intel.analytics.zoo.serving.ClusterServing.{Params, Record}
import com.intel.analytics.zoo.serving.pipeline.{RedisIO, RedisPool}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.util.LongAccumulator
import redis.clients.jedis.{Jedis, JedisPool, JedisPoolConfig}


object InferenceStrategy {
  var params: Params = null
  var bcModel: Broadcast[InferenceModel] = null
  var acc: LongAccumulator = null
  var db: Jedis = null
//  val poolConfig = new JedisPoolConfig()
//  poolConfig.setMaxTotal(128)
  val redisPool = new JedisPool()


  def apply(_params: Params,
            _bcModel: Broadcast[InferenceModel],
            _acc: LongAccumulator,
            r: RDD[(String, Tensor[Float])],
            strategy: String): Unit = {
    params = _params
    bcModel = _bcModel
    acc = _acc


    val result = strategy match {
      case "single" => singleThreadInference(r)
      case "all" => allThreadInference(r)
      case _ => allThreadInference(r)
    }
    result
  }

  /**
   * Use single thread to do inference for one tensor
   * This is used for some BLAS model, such as BigDL
   * Where lower level of parallelism is not supported
   *
   * Throughput will be normal but latency will be high
   */
  def singleThreadInference(r: RDD[(String, Tensor[Float])]): Unit = {
    r.mapPartitions(it => {
      it.grouped(params.coreNum).flatMap(itemBatch => {
        itemBatch.indices.toParArray.map(i => {
          acc.add(1)
          val uri = itemBatch(i)._1
          val t = itemBatch(i)._2
          val localPartitionModel = bcModel.value
          val result = localPartitionModel.doPredict(t.addSingletonDimension()).toTensor[Float]

          val value = PostProcessing(result, params.filter)


        })
      })
    })
  }

  /**
   * Use all thread to do inference for one tensor
   * This is used for most of model, such as Tensorflow
   * Where lower level of parallelism is supported
   */
  def allThreadInference(r: RDD[(String, Tensor[Float])]): Unit = {
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


        db = RedisIO.getRedisClient()
        val ppl = db.pipelined()
        val a = new Array[(String, String)](thisBatchSize)
        (0 until thisBatchSize).toParArray.map(i => {
          val value = PostProcessing(result.select(1, i + 1), params.filter)
          (pathByteBatch(i)._1, value)
        }).copyToArray(a)
        a.foreach(x => {
          RedisIO.writeHashMap(ppl, x._1, x._2)
        })

        ppl.sync()
        db.close()
        None
      })
    }).count()
  }

}
