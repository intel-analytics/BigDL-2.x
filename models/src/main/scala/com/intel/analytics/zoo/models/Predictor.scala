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

package com.intel.analytics.bigdl.zoo.models

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.{SampleToMiniBatch, _}
import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.nn.SpatialShareConvolution

import scala.reflect.ClassTag
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image._
import com.intel.analytics.bigdl.utils.Util._
import com.intel.analytics.bigdl.utils.{Engine, MklBlas, Util}
import com.intel.analytics.zoo.models.Configure

/**
 * Predictor for BigDL models
 * @param model BigDL model
 * @param configure configure includes preprocessor, postprocessor, batch size, label mapping
 *                  models from BigDL model zoo have their default configures
 *                  if you want to predict over your own model, or if you want to change the
 *                  default configure, you can pass in a user-defined configure
 */
class Predictor[T: ClassTag](
  model: AbstractModule[Activity, Activity, T],
  var configure: Configure[T] = null
)(implicit ev: TensorNumeric[T]) {
  SpatialShareConvolution.shareConvolution[T](model)
  configure = if (null == configure) Configure.parse(model.getName()) else configure

  private var localPredictor: LocalPredictor[T] = null
  private var distributedPredictor: DistributedPredictor[T] = null
  /**
   * Model prediction for BigDL model zoo.
   *
   * @param imageFrame local or distributed imageFrame
   * @param outputLayer output layer name, if it is null, use output of last layer
   * @param shareBuffer share buffer of output layer
   * @param predictKey key to store prediction result
   * @return imageFrame with prediction
   */
  def predict(imageFrame: ImageFrame,
    outputLayer: String = null,
    shareBuffer: Boolean = false,
    predictKey: String = ImageFeature.predict): ImageFrame = {

    // apply preprocessing if preProcessor is defined
    val data = if (null != configure.preProcessor) {
      imageFrame -> configure.preProcessor
    } else {
      imageFrame
    }
    val predictor = data match {
      case distributedImageFrame: DistributedImageFrame =>
        if (null == distributedPredictor) distributedPredictor =
          new DistributedPredictor[T](model,
            configure.featurePaddingParam,
            configure.batchPerPartition)
        distributedPredictor
      case localImageFrame: LocalImageFrame =>
        if (null == localPredictor) localPredictor =
          new LocalPredictor[T](model,
            configure.featurePaddingParam,
            configure.batchPerPartition)
        localPredictor
    }


    val result = predictor.predictImage(
      data, outputLayer, shareBuffer, predictKey)

    // apply post process if defined
    if (null != configure.postProcessor) configure.postProcessor(result) else result
  }
}

object Predictor {
  def apply[T: ClassTag](
    model: AbstractModule[Activity, Activity, T],
    configure: Configure[T] = null)(implicit ev: TensorNumeric[T]): Predictor[T] =
    new Predictor(model, configure)


  private[models] def predictImageBatch[T: ClassTag](
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

  private[models] def predictSamples[T: ClassTag]
  (localModel: Module[T], samples: Seq[Sample[T]],
    localToBatch: Transformer[Sample[T], MiniBatch[T]],
    shareBuffer: Boolean,
    outputLayer: String = null)(implicit ev: TensorNumeric[T]): Iterator[Tensor[T]] = {
    localToBatch(samples.toIterator).flatMap(batch => {
      localModel.forward(batch.getInput())
      val output = if (outputLayer == null) {
        localModel.output.toTensor[T]
      } else {
        localModel(outputLayer).get.output.toTensor[T]
      }
      val result = if (shareBuffer) output else output.clone()
      if (result.dim() == 1) {
        Array(result)
      } else {
        result.split(1)
      }
    })
  }
}

trait ImagePredictor extends Serializable {
  def predictImage(imageFrame: ImageFrame,
    outputLayer: String = null,
    shareBuffer: Boolean = false,
    predictKey: String = ImageFeature.predict): ImageFrame
}

/**
 * Predictor for distributed data
 *
 * @param model BigDL model
 * @param featurePaddingParam featurePaddingParam if the inputs have variant size
 * @param batchPerPartition batch size per partition, default is 4
 */
class DistributedPredictor[T: ClassTag] private[models](
  model: Module[T],
  featurePaddingParam: Option[PaddingParam[T]] = None,
  batchPerPartition: Int = 4)
  (implicit ev: TensorNumeric[T]) extends ImagePredictor {
  /**
   * model predict DistributedImageFrame, return imageFrame with predicted tensor
   *
   * @param imageFrame imageFrame that contains images
   * @param outputLayer if outputLayer is not null, the output of layer that matches
   * outputLayer will be used as predicted output
   * @param shareBuffer whether to share same memory for each batch predict results
   * @param predictKey key to store predicted result
   */
  def predictImage(imageFrame: ImageFrame,
    outputLayer: String = null,
    shareBuffer: Boolean = false,
    predictKey: String = ImageFeature.predict): ImageFrame = {
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

/**
 * Predictor for local data
 *
 * @param model BigDL model
 * @param featurePaddingParam featurePaddingParam if the inputs have variant size
 * @param batchPerCore batch size per core, default is 4
 */
class LocalPredictor[T: ClassTag] private[models](model: Module[T],
  featurePaddingParam: Option[PaddingParam[T]] = None,
  batchPerCore: Int = 4)
  (implicit ev: TensorNumeric[T]) extends ImagePredictor {
  private val coreNumber = Engine.coreNumber()

  private val subModelNumber = Engine.getEngineType match {
    case MklBlas => coreNumber
    case _ => throw new IllegalArgumentException
  }

  private val workingModels = {
    val weightsBias = Util.getAndClearWeightBias(model.parameters())
    val models = (1 to subModelNumber).map(_ => {
      val submodel = model.cloneModule().evaluate()
      putWeightBias(weightsBias, submodel)
      submodel
    }).toArray
    Util.putWeightBias(weightsBias, model)
    Util.initGradWeightBias(weightsBias, model)
    models
  }

  val workingToBatch = {
    val toBatch = SampleToMiniBatch[T](
      batchSize = batchPerCore * subModelNumber,
      partitionNum = Some(subModelNumber),
      featurePaddingParam = featurePaddingParam)
    (1 to subModelNumber).map(_ => {
      toBatch.cloneTransformer()
    }).toArray
  }


  /**
   * local model predict images, return imageFrame with predicted tensor
   *
   * @param imageFrame imageFrame that contains images
   * @param outputLayer if outputLayer is not null, the output of layer that matches
   * outputLayer will be used as predicted output
   * @param shareBuffer whether to share same memory for each batch predict results
   * @param predictKey key to store predicted result
   */
  def predictImage(imageFrame: ImageFrame,
    outputLayer: String = null,
    shareBuffer: Boolean = false,
    predictKey: String = ImageFeature.predict): ImageFrame = {

    val dataIter = imageFrame.toLocal().array.grouped(batchPerCore * subModelNumber)

    val result = dataIter.map(batch => {
      val groupedImages = batch.grouped(batchPerCore).toArray
      Engine.default.invokeAndWait(
        groupedImages.indices.map(b =>
          () => {
            val imageFeatures = groupedImages(b)
            val model = workingModels(b)
            val toBatch = workingToBatch(b)
            Predictor.predictImageBatch[T](model, imageFeatures, outputLayer, predictKey,
              toBatch, shareBuffer)
          }
        )
      ).flatten
    }).flatten

    ImageFrame.array(result.toArray)
  }
}
