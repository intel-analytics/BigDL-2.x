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
package com.intel.analytics.zoo.pipeline.api.keras.layers

import java.io.{File, FileInputStream, InputStream}
import java.nio.FloatBuffer

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Shape, T, Table}
import org.tensorflow.framework.GraphDef
import org.tensorflow.{Graph, Session, Tensor => TTensor}

import scala.collection.JavaConverters._

class TFNet(graphDef: GraphDef,
            val inputNames: Seq[String],
            val outputNames: Seq[String])
  extends AbstractModule[Activity, Activity, Float] {

  @transient
  private lazy val graph = {
    val g = new Graph()
    g.importGraphDef(graphDef.toByteArray)
    g
  }

  @transient
  private lazy val sess = {
    new Session(graph)
  }

  override def parameters(): (Array[Tensor[Float]], Array[Tensor[Float]]) = {
    (Array(), Array())
  }

  private def bigdl2Tf(t: Tensor[Float]) = {
    val shape = t.size().map(_.toLong)
    val arr = t.storage().array()
    val buffer = FloatBuffer.wrap(arr)
    TTensor.create(shape, buffer)
  }

  private def tf2bigdl(t: TTensor[Float]) = {
    val shape = t.shape().map(_.toInt)
    val data = new Array[Float](shape.product)
    val buffer = FloatBuffer.wrap(data)
    t.writeTo(buffer)
    Tensor(data, shape)
  }

  override def updateOutput(input: Activity): Activity = {
    val data = input match {
      case t: Tensor[Float] =>
        val tfTensor = bigdl2Tf(t)
        Seq(tfTensor)
      case t: Table =>
        for (i <- 1 to t.length()) yield {
          bigdl2Tf(t[Tensor[Float]](i))
        }
    }

    val runner = sess.runner()
    outputNames.foreach(runner.fetch)
    inputNames.zipWithIndex.foreach { case (name, idx) =>
      runner.feed(name, data(idx))
    }
    val outputs = runner.run()
    val tensors = outputs.asScala.map(t => tf2bigdl(t.asInstanceOf[TTensor[Float]]))
    output = if (tensors.length == 1) {
      tensors.head
    } else {
      T.seq(tensors)
    }
    output
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    throw new Exception("Not Supported Yet")
  }
}

object TFNet {
  def apply(graphDef: GraphDef, inputNames: Seq[String],
            outputNames: Seq[String]): TFNet = {
    new TFNet(graphDef, inputNames, outputNames)
  }

  def apply(path: String, inputNames: Seq[String],
            outputNames: Seq[String]): TFNet = {
    val graphDef = parse(path)
    new TFNet(graphDef, inputNames, outputNames)
  }

  private def parse(graphProtoTxt: String) : GraphDef = {
    var fr: File = null
    var in: InputStream = null
    try {
      fr = new File(graphProtoTxt)
      in = new FileInputStream(fr)

      GraphDef.parseFrom(in)
    } finally {
      if (in != null) in.close()
    }

  }
}
