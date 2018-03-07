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
package com.intel.analytics.bigdl.utils.tf.loaders

import java.nio.ByteOrder

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.ops.{Dilation2DBackpropInput => Dilation2DBackpropInputOps}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.tf.Context
import org.tensorflow.framework.{DataType, NodeDef}

import scala.reflect.ClassTag

class Dilation2DBackpropInput extends TensorflowOpsLoader {

  import Utils._

  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder,
                                  context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    val attributes = nodeDef.getAttrMap
    val padding = getString(attributes, "padding")
    val strides = getIntList(attributes, "strides").toArray
    val rates = getIntList(attributes, "rates").toArray
    val t = getType(nodeDef.getAttrMap, "T")

    if (t == DataType.DT_FLOAT) {
      Dilation2DBackpropInputOps[T, Float](strides, rates, padding)
    } else if (t == DataType.DT_DOUBLE) {
      Dilation2DBackpropInputOps[T, Double](strides, rates, padding)
    } else {
      throw new UnsupportedOperationException(
        s"Not support load Dilation2DBackpropInput when type is ${t}")
    }
  }
}
