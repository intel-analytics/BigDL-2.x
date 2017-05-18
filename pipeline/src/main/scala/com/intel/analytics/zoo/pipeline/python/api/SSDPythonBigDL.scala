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

package com.intel.analytics.bigdl.python.api

import java.lang.{Boolean => JBoolean}
import java.util.{ArrayList => JArrayList, HashMap => JHashMap, List => JList, Map => JMap}

import com.intel.analytics.bigdl.dataset.{Identity => DIdentity, Sample => JSample}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.numeric._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.pipeline.ssd._
import org.apache.spark.api.java.JavaSparkContext

import scala.collection.JavaConverters._
import scala.language.existentials
import scala.reflect.ClassTag

object SSDPythonBigDL {

  def ofFloat(): PythonBigDL[Float] = new SSDPythonBigDL[Float]()

  def ofDouble(): PythonBigDL[Double] = new SSDPythonBigDL[Double]()

}


class SSDPythonBigDL[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonBigDL[T] {

  def ssdPredict(resolution: Int, batchSize: Int, nPartition: Int,
    folder: String, sc: JavaSparkContext,
    model: AbstractModule[Activity, Activity, Float], nClasses: Int = 21)
  : JList[JList[JList[JList[Float]]]] = {
    val predictor = new Predictor(model,
      PreProcessParam(batchSize, resolution, (123f, 117f, 104f), false), nClasses)
    val data = IOUtils.loadLocalFolder(nPartition, folder, sc)
    val results = predictor.predict(data).collect()
    val pathArr = data.map(x => x.path).collect()
    results.zip(pathArr).map(res => {
      val bboxes = res._1.map(r => if (r != null) r.bboxes.storage().array().toList.asJava
      else null).toList.asJava
      val scores = res._1.map(r => if (r != null) r.classes.storage().array().toList.asJava
      else null).toList.asJava
      List(scores, bboxes).asJava
    }).toList.asJava
  }
}



