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

package com.intel.analytics.bigdl.optim

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.{MiniBatch, PaddingParam, Sample, SampleToMiniBatch, Transformer, Utils, DataSet => _}
import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image.{DistributedImageFrame, ImageFeature, ImageFrame}
import com.intel.analytics.bigdl.utils.{T, Table}
import org.apache.spark.rdd.RDD
import Predictor._

import scala.reflect.ClassTag

object Predictor {
  def apply[T: ClassTag](model: Module[T],
    featurePaddingParam: Option[PaddingParam[T]] = None,
    batchPerPartition: Int = 4)
    (implicit ev: TensorNumeric[T]): Predictor[T] = {
    new Predictor[T](model, featurePaddingParam, batchPerPartition)
  }

  private[optim] def predictImageBatch[T: ClassTag](
    localModel: Module[T], imageFeatures: Seq[ImageFeature],
    outputLayer: String, predictKey: String,
    localToBatch: Transformer[Sample[T], MiniBatch[T]],
    shareBuffer: Boolean)(implicit ev: TensorNumeric[T]): Seq[ImageFeature] = {
    val validImageFeatures = imageFeatures.filter(_.isValid)
    val samples = validImageFeatures.map(x => x[Sample[T]](ImageFeature.sample))
    val batchOut = predictSamples(localModel, samples, localToBatch, shareBuffer, outputLayer)
    validImageFeatures.toIterator.zip(batchOut).foreach(tuple => {
      tuple._1(predictKey) = tuple._2
    })
    imageFeatures
  }

  private[optim] def predictSamples[T: ClassTag]
  (localModel: Module[T], samples: Seq[Sample[T]],
    localToBatch: Transformer[Sample[T], MiniBatch[T]],
    shareBuffer: Boolean,
    outputLayer: String = null)(implicit ev: TensorNumeric[T]): Iterator[Activity] = {
    val layer = if (outputLayer == null) {
      localModel
    } else {
      val ol = localModel(outputLayer)
      require(ol.isDefined, s"cannot find layer that map name $outputLayer")
      ol.get
    }
    localToBatch(samples.toIterator).flatMap(batch => {
      localModel.forward(batch.getInput())
      splitBatch[T](layer.output, shareBuffer, batch.size())
    })
  }

  private[optim] def splitBatch[T: ClassTag](output: Activity, shareBuffer: Boolean, batchSize: Int)
    (implicit ev: TensorNumeric[T]): Array[Activity] = {
    val out = if (output.isTensor) {
      val result = if (shareBuffer) output.toTensor[T] else output.toTensor[T].clone()
      if (result.dim() == 1) {
        require(batchSize == 1,
          s"If result dim == 1, the batchSize is required to be 1, while actual is $batchSize")
        Array(result)
      } else {
        result.split(1)
      }
    } else {
      val result = output.toTable
      val first = result[Tensor[T]](1)
      if (first.dim() == 1) {
        require(batchSize == 1,
          s"If result dim == 1, the batchSize is required to be 1, while actual is $batchSize")
        val table = if (shareBuffer) {
          result
        } else {
          val table = T()
          (1 to result.length()).foreach(key => {
            table.insert(result[Tensor[T]](key).clone())
          })
          table
        }
        Array(table)
      } else {
        val batch = first.size(1)
        require(batch == batchSize, s"output batch $batch is not equal to input batch $batchSize")
        val tables = new Array[Table](batch)
        var i = 1
        while (i <= batch) {
          val table = T()
          tables(i - 1) = table
          (1 to result.length()).foreach(key => {
            val split = result[Tensor[T]](key)(i)
            if (shareBuffer) {
              table.insert(split)
            } else {
              table.insert(split.clone())
            }
          })
          i += 1
        }
        tables
      }
    }
    out.asInstanceOf[Array[Activity]]
  }
}

/**
 * Predictor for distributed data
 * @param model BigDL model
 * @param featurePaddingParam featurePaddingParam if the inputs have variant size
 * @param batchPerPartition batch size per partition, default is 4
 */
class Predictor[T: ClassTag] private[optim](
   model: Module[T],
   featurePaddingParam: Option[PaddingParam[T]] = None,
   batchPerPartition: Int = 4)
  (implicit ev: TensorNumeric[T]) extends Serializable {

  def predictClass(dataSet: RDD[Sample[T]], batchSize: Int = -1): RDD[Int] = {
    val result = predict(dataSet, batchSize, true)
    result.mapPartitions { partition =>
      partition.map(output => {
        val _output = output.toTensor[T]
        require(_output.dim() == 1, s"Predictor.predictClass:" +
          s"Only support one sample has one label, but got ${_output.dim()} label")
        ev.toType[Int](_output.max(1)._2.valueAt(1))
      })
    }
  }

  def predict(dataSet: RDD[Sample[T]], batchSize: Int = -1,
              shareBuffer: Boolean = false): RDD[Activity] = {
    val modelBroad = ModelBroadcast[T]().broadcast(dataSet.sparkContext, model.evaluate())
    val partitionNum = dataSet.partitions.length
    val totalBatch = if (batchSize > 0) {
      require(batchSize % partitionNum == 0, s"Predictor.predict: total batch size $batchSize " +
        s"should be divided by partitionNum ${partitionNum}")
      batchSize
    } else {
      batchPerPartition * partitionNum
    }
    val otherBroad = dataSet.sparkContext.broadcast(SampleToMiniBatch(
      batchSize = totalBatch,
      partitionNum = Some(partitionNum),
      featurePaddingParam = featurePaddingParam))
    dataSet.mapPartitions { partition =>
      val localModel = modelBroad.value()
      val localTransformer = otherBroad.value.cloneTransformer()
      val miniBatch = localTransformer(partition)
      miniBatch.flatMap( batch => {
        val output = localModel.forward(batch.getInput)
        splitBatch(output, shareBuffer, batch.size())
      })
    }
  }


  /**
   * model predict DistributedImageFrame, return imageFrame with predicted tensor
   * @param imageFrame imageFrame that contains images
   * @param outputLayer if outputLayer is not null, the output of layer that matches
   *                      outputLayer will be used as predicted output
   * @param shareBuffer whether to share same memory for each batch predict results
   * @param predictKey key to store predicted result
   */
  def predictImage(imageFrame: DistributedImageFrame,
    outputLayer: String = null,
    shareBuffer: Boolean = false,
    predictKey: String = ImageFeature.predict): DistributedImageFrame = {
    val rdd = imageFrame.asInstanceOf[DistributedImageFrame].rdd
    val modelBroad = ModelBroadcast[T]().broadcast(rdd.sparkContext, model.evaluate())
    val partitionNum = rdd.partitions.length
    val toBatchBroad = rdd.sparkContext.broadcast(SampleToMiniBatch(
      batchSize = partitionNum * batchPerPartition,
      partitionNum = Some(partitionNum),
      featurePaddingParam = featurePaddingParam), shareBuffer)
    val result = rdd.mapPartitions(partition => {
      val localModel = modelBroad.value()
      val localToBatch = toBatchBroad.value._1.cloneTransformer()

      partition.grouped(batchPerPartition).flatMap(imageFeatures => {
        Predictor.predictImageBatch[T](localModel, imageFeatures, outputLayer, predictKey,
          localToBatch, shareBuffer)
      })
    })
    ImageFrame.rdd(result)
  }
}
