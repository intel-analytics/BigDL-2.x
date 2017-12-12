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

package com.intel.analytics.zoo.models

import scala.reflect.ClassTag
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image._
import com.intel.analytics.zoo.models.utils.Configure

class Model[A <: Activity : ClassTag, B <: Activity : ClassTag, T: ClassTag]
(model: AbstractModule[A, B, T])(implicit ev: TensorNumeric[T]) {

  def predictFeature(imageFrame: ImageFrame,
    outputLayer: String = null,
    shareBuffer: Boolean = false,
    predictKey: String = ImageFeature.predict): ImageFrame = {
    imageFrame match {
      case distImageFrame: DistributedImageFrame =>
        val config = Configure(model.getName())
        model.predictImage(imageFrame -> config.preProcessor, outputLayer,
          shareBuffer, config.batchPerPartition, predictKey)
      case localImageFrame: LocalImageFrame =>
        throw new NotImplementedError("local predict is not supported for now, coming soon")
    }
  }
}

object Model {
  implicit def abstractModuleToModel[T: ClassTag](model: AbstractModule[Activity, Activity, T])(
    implicit ev: TensorNumeric[T])
  : Model[Activity, Activity, T] = new Model[Activity, Activity, T](model)
}
