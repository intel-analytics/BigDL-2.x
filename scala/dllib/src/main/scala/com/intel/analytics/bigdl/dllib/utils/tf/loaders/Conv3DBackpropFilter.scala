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
import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.nn.tf.{Conv3DBackpropFilter => Conv3DBackpropFilterOps}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.tf.Context
import org.tensorflow.framework.NodeDef

import scala.reflect.ClassTag

class Conv3DBackpropFilter extends TensorflowOpsLoader {

  import Utils._

  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder,
                                  context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    val attributes = nodeDef.getAttrMap
    val (pT, pW, pH) =
      if (getString(attributes, "padding") == "SAME") {
        (-1, -1, -1)
      } else {
        (0, 0, 0)
      }
    val strideList = getIntList(attributes, "strides")
    require(strideList.head == 1, s"not support strides on batch")

    require(strideList(4) == 1, s"not support strides on depth")
    val dT = strideList(1)
    val dW = strideList(2)
    val dH = strideList(3)
    Conv3DBackpropFilterOps[T](dT, dW, dH, pT, pW, pH, DataFormat.NHWC)
  }
}
