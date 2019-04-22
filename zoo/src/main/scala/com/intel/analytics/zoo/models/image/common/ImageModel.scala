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

import com.intel.analytics.bigdl.dataset.SampleToMiniBatch
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.optim.{ValidationMethod, ValidationResult}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.feature.image.ImageSet
import com.intel.analytics.zoo.models.common.ZooModel
import com.intel.analytics.zoo.models.image.imageclassification.ImageClassifier
import com.intel.analytics.zoo.models.image.objectdetection.ObjectDetector
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

  def evaluateImageSet(
     image: ImageSet,
     batchSize: Int,
     vMethods: Array[_ <:ValidationMethod[T]],
     configure: ImageConfigure[T] = null): Array[(ValidationResult, ValidationMethod[T])] = {
    val evalConfig = if (null == configure) config else configure

    if (evalConfig == null) {
      model.evaluateImage(image.toImageFrame(), vMethods, Some(batchSize))
    } else {
      val data = if (null != evalConfig.preProcessor) {
        image -> evalConfig.preProcessor
      } else {
        image
      }
      val dataset = data.toDataSet[T]() -> SampleToMiniBatch[T](batchSize)
      model.evaluate(dataset.toDistributed().data(train = false), vMethods)
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
  def loadModel[T: ClassTag](path: String, weightPath: String = null, modelType: String = "",
                             quantize: Boolean = false)
    (implicit ev: TensorNumeric[T]): ImageModel[T] = {
    var model = Module.loadModule[T](path, weightPath)
    val modelName = model.getName()
    if (quantize) model = model.quantize()
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
