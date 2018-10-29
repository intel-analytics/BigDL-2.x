package com.intel.analytics.zoo.optim

import com.intel.analytics.bigdl.{Criterion, DataSet, Module}
import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.optim.{DistriOptimizer, LocalOptimizer, Optimizer}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import org.apache.spark.rdd.RDD
import org.spark_project.jetty.io.ArrayByteBufferPool.Bucket

import scala.reflect.ClassTag

object OptimizerWrapper {

  def splitRdds[T] (trainRdd: RDD[Sample[T]], bucket:Int) ={

    val buckets = (0 to bucket -1).map(x=> 1.0 / bucket).toArray

    trainRdd.randomSplit(buckets, 1L)

  }


  /**
    * Apply an Optimizer.
    *
    * @param model               model will be optimized
    * @param sampleRDD           training Samples
    * @param criterion           loss function
    * @param batchSize           mini batch size
    * @param featurePaddingParam feature padding strategy, see
    *                            [[com.intel.analytics.bigdl.dataset.PaddingParam]] for details.
    * @param labelPaddingParam   label padding strategy, see
    *                            [[com.intel.analytics.bigdl.dataset.PaddingParam]] for details.
    * @return An optimizer
    */
  def apply[T: ClassTag](
                          model: Module[T],
                          sampleRDD: RDD[Sample[T]],
                          criterion: Criterion[T],
                          batchSize: Int,
                          featurePaddingParam: PaddingParam[T] = null,
                          labelPaddingParam: PaddingParam[T] = null,
                          buckets:Int = 1
                        ) = {

    val opt = Optimizer(model,sampleRDD,criterion,batchSize,featurePaddingParam,labelPaddingParam)

    if(buckets == 1){
      opt.optimize()
    } else {



    }
  }


  /**
    * Apply an optimizer.
    * User can supply a customized implementation of trait MiniBatch to define
    * how data is organize and retrieved in a mini batch.
    *
    * @param model model will be optimized
    * @param sampleRDD training Samples
    * @param criterion loss function
    * @param batchSize mini batch size
    * @param miniBatchImpl An User-Defined MiniBatch implementation
    * @return an new Optimizer
    */
  def apply[T: ClassTag](
                          model: Module[T],
                          sampleRDD: RDD[Sample[T]],
                          criterion: Criterion[T],
                          batchSize: Int,
                          miniBatchImpl: MiniBatch[T]
                        ) = {

  }

  /**
    * Apply an optimizer.
    *
    * @param model model will be optimizied
    * @param dataset the input dataset - determines the type of optimizer
    * @param criterion loss function
    * @return an new Optimizer
    */
  def apply[T: ClassTag, D](
                             model: Module[T],
                             dataset: DataSet[D],
                             criterion: Criterion[T]
                           )= {

  }

}
