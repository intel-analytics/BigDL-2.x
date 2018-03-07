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
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.ops.{Slice => SliceOps}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.tf.Context
import org.tensorflow.framework.NodeDef

import scala.reflect.ClassTag

class Slice extends TensorflowOpsLoader {


  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder,
    context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    new SliceLoadTF[T]()
  }
}

class SliceLoadTF[T: ClassTag]()(implicit ev: TensorNumeric[T]) extends Adapter[T](Array(2, 3)) {
  import Utils._

  override def build(tensorArrays: Array[Tensor[_]]): AbstractModule[Activity, Activity, T] = {
    val size = tensorArrays(1).asInstanceOf[Tensor[Int]]
    SliceOps[T](toArray(tensorArrays(0).asInstanceOf[Tensor[Int]]),
      toArray(tensorArrays(1).asInstanceOf[Tensor[Int]]))
  }
}
