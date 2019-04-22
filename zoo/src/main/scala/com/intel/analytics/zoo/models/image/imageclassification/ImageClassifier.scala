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

package com.intel.analytics.zoo.models.image.imageclassification

import com.intel.analytics.bigdl.nn.Module

import scala.reflect.ClassTag
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.models.image.common.{ImageConfigure, ImageModel}

/**
 * An Image Classification model.
 *
 */
private[zoo] class ImageClassifier[T: ClassTag]
(implicit ev: TensorNumeric[T]) extends ImageModel[T] {

  override def buildModel(): AbstractModule[Activity, Activity, T] = {
    throw new UnsupportedOperationException("imageclassifier class only support load from file")
  }

}

object ImageClassifier {
  /**
   * Load an pre-trained image classification model (with weights).
   *
   * @param path The path for the pre-trained model.
   *             Local file system, HDFS and Amazon S3 are supported.
   *             HDFS path should be like "hdfs://[host]:[port]/xxx".
   *             Amazon S3 path should be like "s3a://bucket/xxx".
   * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
   * @return
   */
  def loadModel[T: ClassTag](path: String, weightPath: String = null, quantize: Boolean = false)
                            (implicit ev: TensorNumeric[T]): ImageClassifier[T] = {
    ImageModel.loadModel(path, weightPath, "imageclassification", quantize)
      .asInstanceOf[ImageClassifier[T]]
  }

}
