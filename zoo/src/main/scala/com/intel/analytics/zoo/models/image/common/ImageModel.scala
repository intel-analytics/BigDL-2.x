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

package com.intel.analytics.zoo.models.image.common

import com.intel.analytics.bigdl.nn.{MklInt8Convertible, Module, SpatialConvolution}
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.optim.{ValidationMethod, ValidationResult}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.feature.image.{DistributedImageSet, ImageSet, LocalImageSet}
import com.intel.analytics.zoo.models.common.ZooModel
import com.intel.analytics.zoo.models.image.imageclassification.ImageClassifier
import com.intel.analytics.zoo.models.image.objectdetection.ObjectDetector
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.MklInt8ConvertibleRef
import org.apache.log4j.Logger

import scala.reflect.ClassTag

/**
 * The base class for image models in Analytics Zoo.
 */
abstract class ImageModel[T: ClassTag]()(implicit ev: TensorNumeric[T])
  extends ZooModel[Activity, Activity, T] {

  private var config: ImageConfigure[T] = null

  /**
   * Computes the output using the current parameter set of the class and input. This function
   * returns the result which is stored in the output field.
   *
   * @param image
   * @return
   */
  def predictImageSet(image: ImageSet, configure: ImageConfigure[T] = null):
  ImageSet = {
    val dataLength = image match {
      case distributedImageSet: DistributedImageSet =>
        distributedImageSet.toDistributed().rdd.partitions.length
      case localImageSet: LocalImageSet =>
        localImageSet.toLocal().array.length
    }

    require(dataLength > 0,
      s"ImageModel.predictImageSet: input is empty, please check your image path.")

    val predictConfig = if (null == configure) config else configure

    val result = if (predictConfig == null) {
      ImageSet.fromImageFrame(model.predictImage(image.toImageFrame()))
    } else {
      // apply preprocessing if preProcessor is defined
      val data = if (null != predictConfig.preProcessor) {
        image -> predictConfig.preProcessor
      } else {
        image
      }

      val imageSet = ImageSet.fromImageFrame(model.predictImage(data.toImageFrame(),
        batchPerPartition = predictConfig.batchPerPartition,
        featurePaddingParam = predictConfig.featurePaddingParam))

      if (null != predictConfig.postProcessor) {
        imageSet -> predictConfig.postProcessor
      }
      else imageSet
    }
    result
  }

  /**
   * Evaluate the ImageSet given validation methods.
   * Currently only DistributedImageSet is supported.
   *
   * @param image DistributedImageSet in which each image should have a label.
   * @param vMethods Array of ValidationMethod to evaluate the ImageSet.
   * @param configure An instance of [[ImageConfigure]]. Default is null and it will
   *                  be pre-defined together with each ImageModel.
   */
  def evaluateImageSet(
      image: DistributedImageSet,
      vMethods: Array[_ <:ValidationMethod[T]],
      configure: ImageConfigure[T] = null): Array[(ValidationResult, ValidationMethod[T])] = {
    val evalConfig = if (null == configure) config else configure
    val numPartitions = image.toDistributed().rdd.partitions.length
    if (evalConfig == null) {
      val batchSize = 4 * numPartitions
      model.evaluateImage(image.toImageFrame(), vMethods, Some(batchSize))
    } else {
      val data = if (null != evalConfig.preProcessor) {
        image -> evalConfig.preProcessor
      } else {
        image
      }
      val batchSize = evalConfig.batchPerPartition * numPartitions
      model.evaluateImage(data.toImageFrame(), vMethods, Some(batchSize))
    }
  }

  def getConfig(): ImageConfigure[T] = config

}

object ImageModel {

  val logger = Logger.getLogger(getClass)
  /**
   * Load an pre-trained image model (with weights).
   *
   * @param path The path for the pre-trained model.
   *             Local file system, HDFS and Amazon S3 are supported.
   *             HDFS path should be like "hdfs://[host]:[port]/xxx".
   *             Amazon S3 path should be like "s3a://bucket/xxx".
   * @param weightPath The path for pre-trained weights if any. Default is null.
   * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
   * @return
   */
  def loadModel[T: ClassTag](path: String, weightPath: String = null, modelType: String = "")
    (implicit ev: TensorNumeric[T]): ImageModel[T] = {
    val labor = Module.loadModule[T](path, weightPath)
    // Calling quantize may not keep the original name. Thus first get modelName here.
    val modelName = labor.getName()
    // If there exists a SpatialConvolution layer in the model that has weight scales,
    // then it should be an int8 model with scales generated.
    val isInt8Model = labor.toGraph().getForwardExecutions().map(_.element)
      .exists(x => x.isInstanceOf[SpatialConvolution[T]] &&
        MklInt8ConvertibleRef.getWeightScalesBuffer(
        x.asInstanceOf[MklInt8Convertible]).nonEmpty)
    val model = if (isInt8Model) {
      logger.info("Loading an int8 convertible model. " +
        "Quantize to an int8 model for better performance")
      labor.quantize()
    } else labor
    val imageModel = if (model.isInstanceOf[ImageModel[T]]) {
      model.asInstanceOf[ImageModel[T]]
    } else {
      val specificModel = modelType.toLowerCase() match {
        case "objectdetection" => new ObjectDetector[T]()
        case "imageclassification" => new ImageClassifier[T]()
        case _ => logger.error(s"model type $modelType is not defined in Analytics zoo." +
          s"Only 'imageclassification' and 'objectdetection' are currently supported.")
          throw new IllegalArgumentException(
            s"model type $modelType is not defined in Analytics zoo." +
              s"Only 'imageclassification' and 'objectdetection' are currently supported.")
      }
      specificModel.addModel(model)
      specificModel.setName(modelName)
    }
    imageModel.config = ImageConfigure.parse(modelName)
    imageModel
  }
}
